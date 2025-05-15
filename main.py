import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict
import os
import logging
import re
import functools

# --- Logging Setup ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "debug").upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOGGING_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("proxy.log")],
)
logger = logging.getLogger(__name__)


import time
from typing import Optional

# In-memory async-aware cache for model free status
_model_free_cache = {}
_MODEL_FREE_CACHE_TTL = 3600  # seconds (1 hour)


async def is_model_free_via_api(model_name: str) -> bool:
    """
    Enhanced free model check: If ':free' is not in the model name,
    query the OpenRouter model endpoints API and check if all pricing fields are zero.
    Returns True if the model is free, False otherwise.
    Uses an in-memory async-aware cache with 1 hour TTL.
    """
    now = time.time()
    cache_entry = _model_free_cache.get(model_name)
    if cache_entry:
        value, timestamp = cache_entry
        if now - timestamp < _MODEL_FREE_CACHE_TTL:
            return value

    endpoint_model_name = model_name
    url = f"https://openrouter.ai/api/v1/models/{endpoint_model_name}/endpoints"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch model endpoint info for {model_name}: {resp.status_code}"
                )
                _model_free_cache[model_name] = (False, now)
                return False
            data = resp.json()
            endpoints = data.get("data", {}).get("endpoints", [])
            if not endpoints:
                logger.warning(
                    f"No endpoints found in OpenRouter API response for {model_name}"
                )
                _model_free_cache[model_name] = (False, now)
                return False
            for ep in endpoints:
                pricing = ep.get("pricing", {})
                # If any pricing field is not "0" or 0, treat as paid
                for v in pricing.values():
                    if isinstance(v, str):
                        if v.strip() != "0":
                            _model_free_cache[model_name] = (False, now)
                            return False
                    elif isinstance(v, (int, float)):
                        if v != 0:
                            _model_free_cache[model_name] = (False, now)
                            return False
            # All pricing fields for all endpoints are zero
            _model_free_cache[model_name] = (True, now)
            return True
    except Exception as e:
        logger.warning(f"Error checking free status for model {model_name}: {e}")
        _model_free_cache[model_name] = (False, now)
        return False


# --- Configuration ---
# Using Pydantic for settings management
from pydantic_settings import BaseSettings
from pydantic import Field
import redis.asyncio as redis


class Settings(BaseSettings):
    or_api_keys: str = Field(..., validation_alias="OR_API_KEYS")
    openrouter_api_base: str = Field(
        "https://openrouter.ai/api/v1", validation_alias="OPENROUTER_API_BASE"
    )
    retry_delay_seconds: float = Field(0.1, validation_alias="RETRY_DELAY_SECONDS")
    rpd_limit: int = Field(
        200, validation_alias="RPD_LIMIT"
    )  # Still useful for reference/display
    rpd_limit_patterns: List[str] = Field(
        default=[
            "Rate limit exceeded: limit_rpd",
            "Rate limit exceeded: free-models-per-day",
        ],
        validation_alias="RPD_LIMIT_PATTERNS",
    )
    rpd_cooldown_hours: int = Field(6, validation_alias="RPD_COOLDOWN_HOURS")
    log_level: str = Field("info", validation_alias="LOG_LEVEL")
    redis_url: str | None = Field(None, validation_alias="REDIS_URL")
    host: str = Field("0.0.0.0", validation_alias="HOST")
    port: int = Field(8000, validation_alias="PORT")
    workers: int = Field(1, validation_alias="WORKERS")

    # Retry settings for error prefixes in 200 OK responses
    retry_error_prefixes: List[str] = Field(
        default=[
            '{"error":{"message":"Provider returned error","code":429',
            '{"error":{"message":"Rate limit exceeded: limit_rpm/google/gemini-2.5-pro-exp-03-25',
        ],
        validation_alias="RETRY_ERROR_PREFIXES",
    )
    retry_buffer_size: int = Field(128, validation_alias="RETRY_BUFFER_SIZE")
    max_retries_on_prefix: int = Field(59, validation_alias="MAX_RETRIES_ON_PREFIX")
    retry_on_prefix_delay_seconds: float = Field(
        1.0, validation_alias="RETRY_ON_PREFIX_DELAY_SECONDS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


settings = Settings()

# --- Logging Setup ---
# Use log level from settings
LOG_LEVEL = settings.log_level.upper()
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOGGING_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("proxy.log")],
)
logger = logging.getLogger(__name__)


class RetryWithErrorPrefixException(Exception):
    """Exception raised when a response contains an error prefix that should trigger a retry."""

    def __init__(self, prefix):
        self.prefix = prefix
        super().__init__(f"Response contains error prefix: {prefix}")


# --- API Key Parsing ---
class APIKey:
    def __init__(self, key: str, name: str = None):
        self.key = key
        self.name = name


# Function to parse API keys from settings
def load_api_keys_from_settings(settings: Settings) -> List[APIKey]:
    """
    Parses API keys from the Settings object (sourced from env var).
    Format: name1:key1,name2:key2,name3:key3
    """
    api_keys = []
    env_keys = settings.or_api_keys

    if not env_keys:
        logger.warning(
            "OR_API_KEYS environment variable not set or empty. No API keys loaded."
        )
        return api_keys

    try:
        key_pairs = env_keys.split(",")
        for pair in key_pairs:
            if ":" not in pair:
                logger.warning(
                    f"Invalid key format (missing name): {pair}. Expected format: name:key"
                )
                continue

            name, key = pair.split(":", 1)  # Split on first colon only
            name = name.strip()
            key = key.strip()

            if not key:
                logger.warning(f"Empty API key for name: {name}")
                continue

            api_keys.append(APIKey(key=key, name=name))
            # Log only the last 4 characters of the key for security
            logger.info(
                f"Loaded API key: {name} (ending in ...{key[-4:] if len(key) >= 4 else 'XXXX'})"
            )
    except Exception as e:
        logger.error(f"Error parsing OR_API_KEYS environment variable: {e}")

    return api_keys


# Load API keys from settings
API_KEYS = load_api_keys_from_settings(settings)

# Constants from settings
OPENROUTER_API_BASE = settings.openrouter_api_base
RETRY_DELAY_SECONDS = settings.retry_delay_seconds
RPD_LIMIT = settings.rpd_limit  # Keep for reference
RPD_LIMIT_PATTERNS = settings.rpd_limit_patterns
RPD_COOLDOWN_HOURS = settings.rpd_cooldown_hours


# Function to parse API keys from environment variable
def load_api_keys_from_env() -> List[APIKey]:
    """
    Parses API keys from the OR_API_KEYS environment variable.
    Format: name1:key1,name2:key2,name3:key3
    """
    api_keys = []
    env_keys = os.environ.get("OR_API_KEYS", "")

    if not env_keys:
        logger.warning(
            "OR_API_KEYS environment variable not set or empty. No API keys loaded."
        )
        return api_keys

    try:
        key_pairs = env_keys.split(",")
        for pair in key_pairs:
            if ":" not in pair:
                logger.warning(
                    f"Invalid key format (missing name): {pair}. Expected format: name:key"
                )
                continue

            name, key = pair.split(":", 1)  # Split on first colon only
            name = name.strip()
            key = key.strip()

            if not key:
                logger.warning(f"Empty API key for name: {name}")
                continue

            api_keys.append(APIKey(key=key, name=name))
            # Log only the last 4 characters of the key for security
            logger.info(
                f"Loaded API key: {name} (ending in ...{key[-4:] if len(key) >= 4 else 'XXXX'})"
            )
    except Exception as e:
        logger.error(f"Error parsing OR_API_KEYS environment variable: {e}")

    return api_keys


# --- Key Management ---
@dataclass
class KeyInfo:
    key: str
    name: str = "Unnamed Key"
    last_used_time: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    enabled: bool = True  # To disable keys if they consistently fail
    total_requests: int = 0  # Total requests made with this key
    rate_limited_until: float = 0.0  # Timestamp until which the key is rate limited
    timestamps: List[float] = field(default_factory=list)  # List of request timestamps

    def get_requests_in_window(self, window_start: float, window_end: float) -> int:
        """Count requests between window_start and window_end timestamps."""
        return sum(1 for ts in self.timestamps if window_start <= ts <= window_end)

    def get_available_requests(self, rpd_limit: int) -> int:
        """Calculate available requests based on 24h sliding window."""
        now = time.time()
        window_start = now - 86400  # 24 hours ago
        used_requests = self.get_requests_in_window(window_start, now)
        return max(1, rpd_limit - used_requests)  # Always leave at least 1 available

    def prune_old_timestamps(self, older_than: float):
        """Remove timestamps older than the specified time."""
        self.timestamps = [ts for ts in self.timestamps if ts > older_than]

    async def is_rate_limited(self, redis_client: redis.Redis | None = None) -> bool:
        """Checks if the key is currently in cooldown."""
        now = time.time()
        if redis_client:
            rate_limited_until_str = await redis_client.hget(
                f"keyinfo:{self.key}", "rate_limited_until"
            )
            if rate_limited_until_str:
                self.rate_limited_until = float(rate_limited_until_str)

        is_limited = now < self.rate_limited_until
        if is_limited:
            cooldown_remaining = self.rate_limited_until - now
            logger.debug(
                f"Key ...{self.key[-4:]} is in cooldown for {cooldown_remaining:.1f} more seconds."
            )
        return is_limited

    async def set_rate_limited(
        self, cooldown_seconds: float, redis_client: redis.Redis | None = None
    ):
        """Marks the key as rate-limited for the specified duration."""
        async with self.lock:
            now = time.time()
            self.rate_limited_until = now + cooldown_seconds
            logger.warning(
                f"Key ...{self.key[-4:]} hit rate limit. Cooldown set for {cooldown_seconds / 3600:.1f} hours (until {time.ctime(self.rate_limited_until)})."
            )
            if redis_client:
                await redis_client.hset(
                    f"keyinfo:{self.key}", "rate_limited_until", self.rate_limited_until
                )

    async def record_usage(self, redis_client: redis.Redis | None = None):
        """Records usage, updates last used time, and tracks request timestamp."""
        async with self.lock:
            now = time.time()
            self.last_used_time = now
            self.total_requests += 1
            self.timestamps.append(now)
            # Prune timestamps older than 24 hours
            self.prune_old_timestamps(now - 86400)

            logger.debug(
                f"Key ...{self.key[-4:]}: total={self.total_requests}, window_requests={len(self.timestamps)}"
            )

            if redis_client:
                # Use pipeline for atomic updates
                async with redis_client.pipeline(transaction=True) as pipe:
                    pipe.hset(f"keyinfo:{self.key}", "last_used_time", now)
                    pipe.hincrby(f"keyinfo:{self.key}", "total_requests", 1)
                    # Store timestamps as a comma-separated string
                    pipe.hset(
                        f"keyinfo:{self.key}",
                        "timestamps",
                        ",".join(map(str, self.timestamps)),
                    )
                    await pipe.execute()


class KeyManager:
    def __init__(
        self,
        api_keys: List[APIKey],
        settings: Settings,
        redis_client: redis.Redis | None = None,
    ):
        self.settings = settings
        self.redis_client = redis_client
        if not api_keys:
            logger.error("No API keys provided to KeyManager. Check OR_API_KEYS.")
            raise ValueError("No API keys provided to KeyManager.")

        self.keys: Dict[str, KeyInfo] = {
            key.key: KeyInfo(key=key.key, name=key.name) for key in api_keys
        }
        logger.info(f"KeyManager initialized with {len(self.keys)} keys.")
        # Don't load from Redis here, do it lazily in get_usable_key or at startup

    async def load_key_states_from_redis(self):
        """Loads key states (enabled, total_requests, rate_limited_until, timestamps) from Redis."""
        if not self.redis_client:
            return
        logger.info("Attempting to load key states from Redis...")
        loaded_count = 0
        for key_str, key_info in self.keys.items():
            try:
                key_data = await self.redis_client.hgetall(f"keyinfo:{key_str}")
                if key_data:
                    async with key_info.lock:
                        key_info.enabled = key_data.get(b"enabled", b"1") == b"1"
                        key_info.total_requests = int(
                            key_data.get(b"total_requests", b"0")
                        )
                        key_info.rate_limited_until = float(
                            key_data.get(b"rate_limited_until", b"0.0")
                        )
                        key_info.last_used_time = float(
                            key_data.get(b"last_used_time", b"0.0")
                        )

                        # Load timestamps
                        timestamps_str = key_data.get(b"timestamps", b"")
                        if timestamps_str:
                            key_info.timestamps = [
                                float(ts)
                                for ts in timestamps_str.decode().split(",")
                                if ts
                            ]
                            # Prune old timestamps on load
                            key_info.prune_old_timestamps(time.time() - 86400)

                        loaded_count += 1
                        logger.debug(
                            f"Loaded state for key ...{key_str[-4:]} from Redis (window_requests={len(key_info.timestamps)})"
                        )
            except Exception as e:
                logger.error(
                    f"Error loading state for key ...{key_str[-4:]} from Redis: {e}"
                )
        if loaded_count > 0:
            logger.info(
                f"Successfully loaded state for {loaded_count} keys from Redis."
            )
        else:
            logger.info(
                "No existing key states found in Redis or Redis not configured."
            )

    async def get_usable_key(self, model_name: str | None = None) -> str:
        """
        Finds a usable key based on cooldown status, available requests in the 24h window,
        and per-minute limits for specific models.
        Keys are sorted by:
        1. Not in cooldown
        2. Most available requests in 24h window
        3. Least recently used (as tie-breaker)
        """
        while True:
            # Filter enabled keys
            valid_keys = [k for k in self.keys.values() if k.enabled]
            now = time.time()

            # Sort keys considering both cooldown and available requests
            sorted_keys = sorted(
                valid_keys,
                key=lambda k: (
                    now
                    < k.rate_limited_until,  # Cooldown keys last (True sorts after False)
                    -k.get_available_requests(
                        self.settings.rpd_limit
                    ),  # Most available requests first (negative for descending)
                    k.last_used_time,  # Least recently used first
                ),
            )

            if not sorted_keys:
                logger.error("No API keys available for the request.")
                raise HTTPException(
                    status_code=503,
                    detail="No API keys available.",
                )

            selected_key_info = None
            # Check keys in sorted order
            for key_info in sorted_keys:
                # Check cooldown status
                is_limited = await key_info.is_rate_limited(self.redis_client)
                if not is_limited:
                    # Check per-minute limit for specific models
                    if (
                        model_name == "google/gemini-2.5-pro-exp-03-25"
                        and now - key_info.last_used_time < 60
                    ):
                        logger.debug(
                            f"Key ...{key_info.key[-4:]} used recently for {model_name}, skipping."
                        )
                        continue  # Skip this key, it was used within the last minute

                    # Check available requests
                    available = key_info.get_available_requests(self.settings.rpd_limit)
                    if available > 1:  # Keep at least 1 request as buffer
                        selected_key_info = key_info
                        break
                    else:
                        logger.debug(
                            f"Key ...{key_info.key[-4:]} has insufficient available requests ({available})"
                        )

            if selected_key_info:
                await selected_key_info.record_usage(self.redis_client)
                logger.debug(
                    f"Selected key ...{selected_key_info.key[-4:]} for model {model_name} (total: {selected_key_info.total_requests}, "
                    f"available: {selected_key_info.get_available_requests(self.settings.rpd_limit)}, "
                    f"last used: {time.ctime(selected_key_info.last_used_time)})"
                )
                return selected_key_info.key

            # All keys are either rate-limited, have insufficient available requests, or were used recently for this model
            logger.warning(
                f"All {len(valid_keys)} enabled keys are currently unavailable (cooldown, insufficient requests, or recently used for {model_name}). "
                f"Waiting {self.settings.retry_delay_seconds}s..."
            )
            await asyncio.sleep(self.settings.retry_delay_seconds)

    async def update_key_state(
        self,
        key: str,
        enabled: bool | None = None,
        rate_limited_until: float | None = None,
    ):
        """Updates the state of a key both in memory and optionally in Redis."""
        if key not in self.keys:
            return

        key_info = self.keys[key]
        updated = False
        redis_updates = {}

        async with key_info.lock:
            if enabled is not None and key_info.enabled != enabled:
                key_info.enabled = enabled
                redis_updates["enabled"] = "1" if enabled else "0"
                logger.info(
                    f"{'Enabled' if enabled else 'Disabled'} key ending in ...{key[-4:]}"
                )
                updated = True
            if (
                rate_limited_until is not None
                and key_info.rate_limited_until != rate_limited_until
            ):
                key_info.rate_limited_until = rate_limited_until
                redis_updates["rate_limited_until"] = str(rate_limited_until)
                # Logging is handled in set_rate_limited
                updated = True

        if updated and self.redis_client and redis_updates:
            try:
                await self.redis_client.hset(f"keyinfo:{key}", mapping=redis_updates)
                logger.debug(
                    f"Updated Redis state for key ...{key[-4:]}: {redis_updates}"
                )
            except Exception as e:
                logger.error(f"Failed to update Redis state for key ...{key[-4:]}: {e}")

    async def disable_key(self, key: str):
        """Disables a key."""
        await self.update_key_state(key, enabled=False)

    async def enable_key(self, key: str):
        """Re-enables a key and resets its cooldown."""
        await self.update_key_state(key, enabled=True, rate_limited_until=0.0)


# --- Globals ---
# --- Globals ---
app = FastAPI(title="OpenRouter Reverse Proxy")

# Initialize Redis client if URL is provided
redis_client = None
if settings.redis_url:
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        logger.info(f"Redis client initialized for URL: {settings.redis_url}")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}. Running in-memory only.")
        redis_client = None  # Ensure it's None if init fails

key_manager = KeyManager(API_KEYS, settings, redis_client)  # Pass settings and client

# Use a single httpx client for connection pooling
http_client = httpx.AsyncClient(timeout=None)


# --- Helper Functions ---
async def _forward_request(
    request: Request, target_url: str, api_key: str, body: bytes
):
    """Forwards the incoming request to the target URL with the given API key."""
    headers = dict(request.headers)
    headers.pop("host", None)
    # headers.pop("authorization", None) # Keep client auth if needed for non-free models
    # Required headers for OpenRouter API (https://openrouter.ai/docs)
    # Remove any existing Authorization headers (case-insensitive)
    headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}

    # For paid model requests (client-provided key), api_key is already the correct value.
    # For free model requests, api_key is from our pool.
    headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",  # Required: your site URL
            "Host": "openrouter.ai",  # Required for requests
        }
    )
    # Remove headers that will be set by httpx
    headers.pop("Content-Length", None)
    headers.pop("Transfer-Encoding", None)
    headers.pop("Connection", None)

    # Log headers without sensitive information
    # Log all outgoing headers, including Authorization, for debugging paid model issues
    logger.debug(f"Outgoing headers to OpenRouter: {headers}")
    safe_headers = {k: v for k, v in headers.items() if "auth" not in k.lower()}
    logger.debug(f"Request headers (safe): {safe_headers}")

    # Use the body provided as a parameter instead of reading it again
    content = body

    # Convert retry_error_prefixes to bytes for comparison
    retry_error_prefixes_bytes = [
        prefix.encode() for prefix in settings.retry_error_prefixes
    ]

    try:
        req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=content,
        )
        response = await http_client.send(req, stream=True)  # stream=True is crucial
        logger.debug(f"Response status code: {response.status_code}")

        # Check for specific OpenRouter errors *before* raising for status
        if response.status_code == 401:  # Unauthorized - potentially bad key
            logger.error(
                f"Received 401 Unauthorized for key ...{api_key[-4:]}. Consider disabling it."
            )
            # Optionally disable the key here: key_manager.disable_key(api_key)
            response.raise_for_status()
        elif response.status_code == 402:  # Payment Required
            logger.warning(
                f"Received 402 Payment Required for key ...{api_key[-4:]}. Key might be out of credits."
            )
            response.raise_for_status()
        elif response.status_code == 429:  # Rate Limited by OpenRouter
            logger.warning(f"Rate limited by OpenRouter for key ...{api_key[-4:]}")
            response.raise_for_status()
        else:
            response.raise_for_status()  # Raise for other 4xx/5xx errors

        # For 200 OK responses, buffer initial data to check for retry error prefixes
        if response.status_code == 200:
            # Buffer for detecting error prefixes
            buffer_size = settings.retry_buffer_size
            buffer = b""

            # Create an iterator from the response
            response_iter = response.aiter_bytes()

            # Read up to buffer_size bytes (or less if the response is smaller)
            try:
                for _ in range(buffer_size):
                    try:
                        chunk = await anext(response_iter)
                        buffer += chunk

                        # Check if buffer contains any error prefix (might have newlines before the error)
                        for prefix_bytes in retry_error_prefixes_bytes:
                            if prefix_bytes in buffer:
                                logger.warning(
                                    f"Detected error prefix in stream with 200 OK: {prefix_bytes.decode(errors='ignore')}"
                                )
                                raise RetryWithErrorPrefixException(
                                    prefix_bytes.decode(errors="ignore")
                                )
                    except StopAsyncIteration:
                        # End of response reached before buffer filled
                        break
            except RetryWithErrorPrefixException:
                # Let this propagate up to the calling function
                raise

        # Store headers before streaming starts
        response_headers = dict(response.headers)

        # Create an async generator to check for RPD limit while streaming
        async def check_rpd_stream():
            # First yield our already-read buffer
            if buffer:
                yield buffer

            # Now continue with the rest of the response
            buffer_for_rpd = b""
            max_buffer_size = 128
            async for chunk in response_iter:
                buffer_for_rpd += chunk
                # Try to decode and check for RPD limit in accumulated buffer
                try:
                    text = buffer_for_rpd.decode(
                        errors="ignore"
                    )  # Ignore decode errors for partial data
                    # Check if any RPD limit pattern is in the decoded text
                    if any(pattern in text for pattern in settings.rpd_limit_patterns):
                        logger.warning(
                            f"RPD limit pattern detected in stream for key ...{api_key[-4:]}. Applying cooldown."
                        )
                        key_info = key_manager.keys.get(api_key)
                        if key_info:
                            cooldown_seconds = settings.rpd_cooldown_hours * 3600
                            await key_info.set_rate_limited(
                                cooldown_seconds, key_manager.redis_client
                            )
                            # No need to raise an error here, let the stream continue but mark the key
                        # Clear buffer after match
                        buffer_for_rpd = b""
                    elif len(buffer_for_rpd) > max_buffer_size:
                        # Keep last 1KB in case pattern spans chunks
                        buffer_for_rpd = buffer_for_rpd[-1024:]
                except UnicodeDecodeError:
                    # If we can't decode yet, continue accumulating
                    if len(buffer_for_rpd) > max_buffer_size:
                        # Keep last 1KB even if decode fails
                        buffer_for_rpd = buffer_for_rpd[-1024:]
                yield chunk

        # Stream the response back with RPD checking
        return StreamingResponse(
            check_rpd_stream(),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    except RetryWithErrorPrefixException as e:
        # Close the response to ensure resources are released
        await response.aclose()

        # Log the error prefix detection
        logger.warning(
            f"Detected error prefix '{e.prefix}' in 200 OK response. This response will be retried."
        )

        # Register a "hit" for rate limiting but don't apply cooldown
        key_info = key_manager.keys.get(api_key)
        if key_info:
            # Only record usage, don't set rate_limited_until
            async with key_info.lock:
                key_info.total_requests += 1
                now = time.time()
                key_info.timestamps.append(now)
                key_info.prune_old_timestamps(now - 86400)

                if key_manager.redis_client:
                    # Use pipeline for atomic updates
                    async with key_manager.redis_client.pipeline(
                        transaction=True
                    ) as pipe:
                        pipe.hincrby(f"keyinfo:{api_key}", "total_requests", 1)
                        pipe.hset(
                            f"keyinfo:{api_key}",
                            "timestamps",
                            ",".join(map(str, key_info.timestamps)),
                        )
                        await pipe.execute()

        # Raise the exception to be handled by the caller
        raise

    except httpx.HTTPStatusError as e:
        # Log the status code from the error response
        logger.error(
            f"HTTP Status Error: {e.response.status_code} for URL {e.request.url} with key ...{api_key[-4:]}"
        )

        # Read error content once
        error_content = None
        try:
            error_content = await e.response.aread()
            error_text = error_content.decode()

            # Check for RPD limit in error response text
            if any(pattern in error_text for pattern in settings.rpd_limit_patterns):
                logger.warning(
                    f"RPD limit pattern detected in error response for key ...{api_key[-4:]}. Applying cooldown."
                )
                key_info = key_manager.keys.get(api_key)
                if key_info:
                    cooldown_seconds = settings.rpd_cooldown_hours * 3600
                    await key_info.set_rate_limited(
                        cooldown_seconds, key_manager.redis_client
                    )
                    # Don't re-record usage here, it was already recorded before the request

            # For 400 errors, log the error message
            if e.response.status_code == 400:
                logger.error(f"OpenRouter Error: {error_text}")
        except Exception as decode_err:
            logger.error(f"Failed to decode error response: {decode_err}")
            error_content = b'{"error": "Failed to decode error response"}'

        return StreamingResponse(
            iter([error_content]),
            status_code=e.response.status_code,
            headers=dict(e.response.headers),
            media_type=e.response.headers.get(
                "content-type", "application/json"
            ),  # Default media type
        )
    except httpx.TimeoutException as e:
        logger.error(f"Request timed out: {e}")
        raise HTTPException(
            status_code=504, detail="Gateway Timeout: Upstream request timed out."
        )
    except httpx.RequestError as e:
        logger.error(f"Request Error: {e}")
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {e}")
    except Exception as e:
        logger.exception(f"Unexpected Internal Server Error: {e}")  # Log stack trace
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


templates = Jinja2Templates(directory="templates")


@app.get("/stats", include_in_schema=False)
@app.get("/api/stats", response_class=Response, include_in_schema=False)
async def get_stats(request: Request):
    """Returns API key statistics in HTML, JSON, or plain text format based on Accept header"""
    stats = []
    now = time.time()  # Use Unix timestamp for consistency
    accept = request.headers.get("accept", "")
    want_html = "text/html" in accept
    want_json = request.url.path == "/api/stats" or "application/json" in accept

    for key_info in key_manager.keys.values():
        is_limited = now < key_info.rate_limited_until
        cooldown_remaining = key_info.rate_limited_until - now if is_limited else 0
        available_requests = key_info.get_available_requests(RPD_LIMIT)

        stat_entry = {
            "name": key_info.name,
            "key_suffix": key_info.key[-4:],
            "enabled": key_info.enabled,
            "total_requests": key_info.total_requests,
            "available_requests": available_requests,
            "is_limited": is_limited,
            "cooldown_remaining_seconds": cooldown_remaining,
            "rate_limited_until_str": (
                time.ctime(key_info.rate_limited_until) if is_limited else "N/A"
            ),
            "last_used_str": (
                time.ctime(key_info.last_used_time)
                if key_info.last_used_time > 0
                else "Never"
            ),
            "timestamps": key_info.timestamps,  # Add timestamps for chart
        }
        stats.append(stat_entry)

    if want_json:
        return Response(
            content=json.dumps(
                {
                    "stats": stats,
                    "rpd_limit": RPD_LIMIT,
                    "rpd_cooldown_hours": RPD_COOLDOWN_HOURS,
                }
            ),
            media_type="application/json",
        )
    elif not want_html:
        # ASCII table format for terminal
        ascii_table = [
            "\n" + "═" * 110,  # Top border
            " API KEY MONITORING DASHBOARD ".center(110),
            "═" * 110 + "\n",  # Bottom border of title
        ]

        headers = [
            "NAME",
            "KEY",
            "STATUS",
            "TOTAL",
            "AVAILABLE",
            "COOLDOWN UNTIL",
            "LAST USED",
        ]
        widths = [20, 8, 18, 8, 10, 25, 25]  # Adjusted widths

        # Header
        header_row = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
        ascii_table.append(header_row)
        ascii_table.append("─" * 120)  # Separator (increased width)

        # Data rows
        for row in stats:
            status_text = ""
            if not row["enabled"]:
                status_text = "\033[91m✗ DISABLED\033[0m"  # Red
            elif row["is_limited"]:
                remaining_h = row["cooldown_remaining_seconds"] / 3600
                status_text = (
                    f"\033[93m⏳ COOLING ({remaining_h:.1f}h)\033[0m"  # Yellow
                )
            else:
                status_text = "\033[92m✓ ACTIVE\033[0m"  # Green

            cooldown_until = row["rate_limited_until_str"]

            # Construct the row with proper spacing
            formatted_row = [
                f"{row['name'][: widths[0] - 1]:<{widths[0]}}",
                f"...{row['key_suffix']:<{widths[1]}}",
                f"{status_text:<{widths[2] + 9}}",  # +9 for ANSI codes
                f"{row['total_requests']:<{widths[3]}}",
                f"{row['available_requests']:<{widths[4]}}",
                f"{cooldown_until:<{widths[5]}}",
                f"{row['last_used_str']:<{widths[6]}}",
            ]
            ascii_table.append("  ".join(formatted_row))

        ascii_table.append("\n" + "═" * 120)  # Bottom border (increased width)
        total_reqs = sum(r["total_requests"] for r in stats)
        active_keys = sum(1 for r in stats if r["enabled"] and not r["is_limited"])
        enabled_keys = sum(1 for r in stats if r["enabled"])
        total_available = sum(
            r["available_requests"]
            for r in stats
            if r["enabled"] and not r["is_limited"]
        )
        ascii_table.append(f" Total Requests: {total_reqs}")
        ascii_table.append(f" Active Keys: {active_keys}/{enabled_keys} enabled")
        ascii_table.append(f" Total Available Requests: {total_available}")
        ascii_table.append("═" * 120 + "\n")  # Bottom border (increased width)

        return Response(content="\n".join(ascii_table), media_type="text/plain")

    else:
        # Convert stats to JSON for the chart
        stats_json = json.dumps(
            [{"name": s["name"], "timestamps": s["timestamps"]} for s in stats]
        )

        # HTML format
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "stats": stats,
                "stats_json": stats_json,  # Add timestamp data for chart
                "rpd_limit": RPD_LIMIT,
                "rpd_cooldown_hours": RPD_COOLDOWN_HOURS,
            },
        )


@app.api_route(
    "/api/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def proxy_openrouter(request: Request, path: str):
    """
    Catch-all route to proxy requests to the OpenRouter API.
    Selects a least recently used key that is not rate-limited.
    Applies cooldown if rate limit errors occur.
    Implements retry mechanism for streaming errors.
    """
    body = await request.body()
    model_name = None
    use_proxy_keys = True  # Default to using proxy keys
    client_key = None
    client_auth_header = request.headers.get("authorization")

    # --- Parallel streaming helper for Gemini ---
    async def parallel_streaming_proxy(request, target_url, body, max_parallel=3):
        """
        Launches up to max_parallel streaming requests in parallel, each with a different available key.
        Keeps retrying with new sets of keys until one succeeds.
        """
        retry_delay = 1.0  # seconds between retries if all fail
        attempt = 0
        while True:
            attempt += 1
            logger.info(
                f"[Gemini Parallel Proxy] Attempt {attempt}: launching up to {max_parallel} parallel requests."
            )
            # Gather up to max_parallel available keys (not rate-limited, enabled)
            usable_keys = []
            checked_keys = set()
            for _ in range(max_parallel * 2):  # Try a bit more in case of cooldowns
                try:
                    k = await key_manager.get_usable_key()
                    if k not in checked_keys:
                        usable_keys.append(k)
                        checked_keys.add(k)
                    if len(usable_keys) >= max_parallel:
                        break
                except Exception:
                    break
            if not usable_keys:
                logger.warning(
                    "[Gemini Parallel Proxy] No usable API keys found, waiting before retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue

            # For each key, launch a streaming request coroutine
            tasks = []
            results = [None] * len(usable_keys)
            done_event = asyncio.Event()

            async def stream_attempt(idx, api_key):
                try:
                    headers = dict(request.headers)
                    headers.pop("host", None)
                    headers = {
                        k: v for k, v in headers.items() if k.lower() != "authorization"
                    }
                    headers.update(
                        {
                            "Authorization": f"Bearer {api_key}",
                            "HTTP-Referer": "http://localhost",
                            "Host": "openrouter.ai",
                        }
                    )
                    headers.pop("Content-Length", None)
                    headers.pop("Transfer-Encoding", None)
                    headers.pop("Connection", None)
                    retry_error_prefixes_bytes = [
                        prefix.encode() for prefix in settings.retry_error_prefixes
                    ]
                    req = http_client.build_request(
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        content=body,
                    )
                    response = await http_client.send(req, stream=True)
                    if response.status_code != 200:
                        await response.aclose()
                        results[idx] = (False, None, None)
                        return

                    # Buffer for error prefix detection
                    buffer_size = settings.retry_buffer_size
                    buffer = b""
                    response_iter = response.aiter_bytes()
                    try:
                        for _ in range(buffer_size):
                            try:
                                chunk = await anext(response_iter)
                                buffer += chunk
                                for prefix_bytes in retry_error_prefixes_bytes:
                                    if prefix_bytes in buffer:
                                        await response.aclose()
                                        results[idx] = (False, None, None)
                                        return
                            except StopAsyncIteration:
                                break
                    except Exception:
                        await response.aclose()
                        results[idx] = (False, None, None)
                        return

                    # If we get here, this stream is a candidate
                    if not done_event.is_set():
                        done_event.set()
                        # Prepare StreamingResponse
                        response_headers = dict(response.headers)

                        async def merged_stream():
                            if buffer:
                                yield buffer
                            async for chunk in response_iter:
                                yield chunk
                            await response.aclose()

                        results[idx] = (
                            True,
                            StreamingResponse(
                                merged_stream(),
                                status_code=response.status_code,
                                headers=response_headers,
                                media_type=response.headers.get("content-type"),
                            ),
                            response,
                        )
                    else:
                        await response.aclose()
                        results[idx] = (False, None, None)
                except Exception:
                    results[idx] = (False, None, None)

            # Launch all attempts
            for i, k in enumerate(usable_keys):
                tasks.append(asyncio.create_task(stream_attempt(i, k)))
            # Wait for first to succeed, or all to finish
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # Check if any succeeded
            found_success = False
            for r in results:
                if r is None:
                    continue
                ok, resp, _ = r
                if ok and resp is not None:
                    # Cancel all unfinished tasks (except the winner)
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    return resp
            # If none succeeded, wait and retry
            logger.warning(
                "[Gemini Parallel Proxy] All parallel attempts failed, retrying with new keys after delay..."
            )
            await asyncio.sleep(retry_delay)

    # Determine model type and key strategy only for chat completions
    if path == "chat/completions":
        try:
            data = json.loads(body)
            logger.debug("Request type: chat/completions")

            # Basic validation
            if (
                "messages" not in data
                or not isinstance(data["messages"], list)
                or not data["messages"]
            ):
                raise HTTPException(
                    status_code=400, detail="Messages must be a non-empty array"
                )
            if "model" not in data:
                raise HTTPException(
                    status_code=400, detail="Missing required field: model"
                )

            model_name = data.get("model", "")
            use_proxy_keys = False
            # Enhanced free model check: If ":free" not in name, check OpenRouter API for 0 pricing
            if ":free" in model_name:
                use_proxy_keys = True
                logger.debug(f"Detected free model request: {model_name}")
            else:
                # Enhanced: Check OpenRouter API for 0 pricing
                if await is_model_free_via_api(model_name):
                    use_proxy_keys = True
                    logger.debug(
                        f"Detected free model via OpenRouter API: {model_name}"
                    )
                else:
                    use_proxy_keys = False
                    logger.debug(f"Detected non-free model request: {model_name}")
                    if client_auth_header and client_auth_header.lower().startswith(
                        "bearer "
                    ):
                        client_key = client_auth_header.split(None, 1)[1].strip()
                        if not client_key:
                            logger.warning(
                                f"Empty Bearer token provided for non-free model request: {model_name}"
                            )
                            raise HTTPException(
                                status_code=401,
                                detail="Invalid Authorization header: Bearer token is empty.",
                            )
                        logger.info(
                            f"Using client-provided key for non-free model {model_name}"
                        )
                    else:
                        logger.warning(
                            f"Missing or invalid Authorization header for non-free model request: {model_name}"
                        )
                        raise HTTPException(
                            status_code=401,
                            detail="Authorization header with Bearer token is required for non-free models.",
                        )

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    else:
        # For non-chat paths, assume proxy keys should be used (original behavior)
        logger.debug(
            f"Request path '{path}' is not chat/completions, using proxy keys by default."
        )
        use_proxy_keys = True

    # --- Construct Target URL ---
    target_url = f"{settings.openrouter_api_base}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    logger.debug(f"Target URL: {target_url}")

    # --- Request Forwarding ---
    # Special case: Gemini parallel streaming
    if use_proxy_keys and model_name == "google/gemini-2.5-pro-exp-03-25":
        return await parallel_streaming_proxy(request, target_url, body, max_parallel=3)

    if use_proxy_keys:
        # Free model or non-chat path: Use key manager and retry logic
        max_retries = settings.max_retries_on_prefix
        retry_count = 0
        retry_delay = settings.retry_on_prefix_delay_seconds

        while True:
            selected_key = None  # Ensure selected_key is defined in this scope
            try:
                # Get a key that is enabled and not in cooldown
                selected_key = await key_manager.get_usable_key()
                # Usage is recorded within get_usable_key now

                key_suffix = (
                    f"...{selected_key[-4:]}"
                    if selected_key and len(selected_key) >= 4
                    else "UNKNOWN"
                )
                log_prefix = (
                    f"Retry {retry_count}/{max_retries}: " if retry_count > 0 else ""
                )
                logger.info(
                    f"{log_prefix}Forwarding {request.method} request to {target_url} using proxy key ending in {key_suffix}"
                )

                # Forward the request with the body we already read
                return await _forward_request(request, target_url, selected_key, body)

            except RetryWithErrorPrefixException as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(
                        f"Exceeded maximum retries ({max_retries}) for error prefix detection. Last prefix: {e.prefix}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Upstream provider error: Maximum retry attempts exceeded",
                    )
                logger.warning(
                    f"Retry {retry_count}/{max_retries}: Error prefix detected '{e.prefix}'. Waiting {retry_delay}s before retry."
                )
                await asyncio.sleep(retry_delay)
                continue  # Continue the loop to retry

            except HTTPException:
                raise  # Re-raise HTTP exceptions from _forward_request or key_manager
            except Exception as e:
                # Catch potential errors during key selection or other unexpected issues
                logger.critical(
                    f"Failed to get/use a proxy API key or forward request: {e}",
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Service Unavailable: Error processing request with proxy key.",
                )
    else:
        # Paid model: Forward directly using client's key (already in client_key)
        try:
            # No retry loop here for client keys initially.
            # If needed, specific retry logic for client keys could be added.
            logger.info(
                f"Forwarding {request.method} request to {target_url} using client-provided key"
            )
            # Ensure client_key is not None (should be guaranteed by logic above, but belt-and-suspenders)
            if not client_key:
                logger.error(
                    "Internal error: client_key is None despite use_proxy_keys being False."
                )
                raise HTTPException(
                    status_code=500, detail="Internal server configuration error."
                )

            return await _forward_request(request, target_url, client_key, body)
        except HTTPException:
            raise  # Re-raise HTTP exceptions from _forward_request
        except Exception as e:
            logger.error(
                f"Error forwarding request with client key: {e}", exc_info=True
            )
            # Don't expose detailed internal errors for client key issues
            raise HTTPException(
                status_code=502,
                detail="Bad Gateway: Error communicating with upstream service using client key.",
            )


# --- Lifecycle Events ---
@app.on_event("startup")
async def app_startup():
    """Load key states from Redis if configured."""
    logger.info("API proxy server starting up...")
    if key_manager.redis_client:
        await key_manager.load_key_states_from_redis()
    else:
        logger.info("Redis not configured. Key states are in-memory only.")

    # Log API key configuration on startup
    key_count = len(key_manager.keys)
    if key_count == 0:
        logger.warning("No API keys loaded. Set the OR_API_KEYS environment variable.")
    else:
        logger.info(f"Successfully loaded {key_count} API key(s).")


@app.on_event("shutdown")
async def app_shutdown():
    """Close the httpx client gracefully on shutdown."""
    logger.info("Shutting down...")
    await http_client.aclose()
    if key_manager.redis_client:
        await key_manager.redis_client.close()
        logger.info("Redis client closed.")


# --- Main Execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting OpenRouter Proxy Server...")

    key_count = len(key_manager.keys)
    if key_count == 0:
        logger.warning("No API keys loaded. Set the OR_API_KEYS environment variable")
    else:
        logger.info(f"Managing {key_count} API key(s)")

    # Use settings loaded by Pydantic
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=False,  # Should generally be False for production
    )
