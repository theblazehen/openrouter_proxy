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

# --- Logging Setup ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
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
    workers: int = Field(4, validation_alias="WORKERS")

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

    async def get_usable_key(self) -> str:
        """
        Finds a usable key based on both cooldown status and available requests in the 24h window.
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
                    f"Selected key ...{selected_key_info.key[-4:]} (total: {selected_key_info.total_requests}, "
                    f"available: {selected_key_info.get_available_requests(self.settings.rpd_limit)}, "
                    f"last used: {time.ctime(selected_key_info.last_used_time)})"
                )
                return selected_key_info.key

            # All keys are either rate-limited or have insufficient available requests
            logger.warning(
                f"All {len(valid_keys)} enabled keys are currently unavailable (cooldown or insufficient requests). "
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
async def _forward_request(request: Request, target_url: str, api_key: str):
    """Forwards the incoming request to the target URL with the given API key."""
    headers = dict(request.headers)
    headers.pop("host")
    headers.pop("authorization", None)
    # Required headers for OpenRouter API (https://openrouter.ai/docs)
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
    safe_headers = {k: v for k, v in headers.items() if "auth" not in k.lower()}
    logger.debug(f"Request headers: {safe_headers}")
    content = await request.body()

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

        # Store headers before streaming starts
        response_headers = dict(response.headers)

        # Create an async generator to check for RPD limit while streaming
        async def check_rpd_stream():
            buffer = b""
            max_buffer_size = 16384  # 16KB max buffer size
            async for chunk in response.aiter_bytes():
                buffer += chunk
                # Try to decode and check for RPD limit in accumulated buffer
                try:
                    text = buffer.decode(
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
                        buffer = b""
                    elif len(buffer) > max_buffer_size:
                        # Keep last 1KB in case pattern spans chunks
                        buffer = buffer[-1024:]
                except UnicodeDecodeError:
                    # If we can't decode yet, continue accumulating
                    if len(buffer) > max_buffer_size:
                        # Keep last 1KB even if decode fails
                        buffer = buffer[-1024:]
                yield chunk

        # Stream the response back with RPD checking
        return StreamingResponse(
            check_rpd_stream(),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

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
                f"{row['name'][:widths[0]-1]:<{widths[0]}}",
                f"...{row['key_suffix']:<{widths[1]}}",
                f"{status_text:<{widths[2]+9}}",  # +9 for ANSI codes
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
    """
    # Read and validate the request body
    body = await request.body()

    # For chat completions, ensure required fields are present
    if path == "chat/completions":
        try:
            data = json.loads(body)
            # Log request data for debugging (excluding sensitive content)
            logger.debug("Request type: chat/completions")

            # Ensure required fields
            if "messages" not in data:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required field: messages",
                )
            if not isinstance(data["messages"], list) or not data["messages"]:
                raise HTTPException(
                    status_code=400,
                    detail="Messages must be a non-empty array",
                )
            if "model" not in data:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required field: model",
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in request body",
            )

    # is_completion = "chat/completion" in path # No longer needed for tracking
    try:
        # Get a key that is enabled and not in cooldown
        selected_key = await key_manager.get_usable_key()
        # Usage is recorded within get_usable_key now
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.critical(f"Failed to get a usable API key: {e}")
        raise HTTPException(
            status_code=503, detail="Service Unavailable: No API keys available."
        )

    target_url = f"{settings.openrouter_api_base}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    logger.debug(f"Target URL: {target_url}")

    # Safer logging with check for None
    key_suffix = (
        f"...{selected_key[-4:]}"
        if selected_key and len(selected_key) >= 4
        else "UNKNOWN"
    )
    logger.info(
        f"Forwarding {request.method} request to {target_url} using key ending in {key_suffix}"
    )

    return await _forward_request(request, target_url, selected_key)


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
