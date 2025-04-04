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
class APIKey:
    def __init__(self, key: str, name: str = None):
        self.key = key
        self.name = name


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


# Load API keys from environment
API_KEYS = load_api_keys_from_env()

OPENROUTER_API_BASE = os.environ.get(
    "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
)
# Time in seconds to wait before retrying if all keys are rate-limited
RETRY_DELAY_SECONDS = 0.1
# Default rate limit if fetching fails or isn't implemented yet
# RPD limit - 200 requests per day
RPD_LIMIT = 200
# RPD limit error patterns
RPD_LIMIT_PATTERNS = [
    "Rate limit exceeded: limit_rpd",
    "Rate limit exceeded: free-models-per-day",
]


# --- Key Management ---
@dataclass
class KeyInfo:
    key: str
    name: str = "Unnamed Key"
    last_used_time: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    enabled: bool = True  # To disable keys if they consistently fail
    total_requests: int = 0  # Total requests made with this key
    daily_requests: List[float] = field(
        default_factory=list
    )  # Timestamps of daily requests

    async def is_rate_limited(self) -> bool:
        """Checks if the key has reached its daily completion limit (200 RPD)."""
        now = time.time()
        # Remove requests older than 24 hours
        day_ago = now - 86400
        self.daily_requests = [t for t in self.daily_requests if t > day_ago]
        is_limited = len(self.daily_requests) >= RPD_LIMIT
        if is_limited:
            logger.debug(f"Key ...{self.key[-4:]} hit RPD limit ({RPD_LIMIT}/day)")
        return is_limited

    async def record_usage(self, is_completion: bool = False):
        """Records a request against the rate limit counters."""
        async with self.lock:
            now = time.time()
            self.last_used_time = now
            self.total_requests += 1

            # Record completion if this is a completion request
            if is_completion:
                self.daily_requests.append(now)

            logger.debug(
                f"Key ...{self.key[-4:]}: total={self.total_requests}, completions={len(self.daily_requests)}/{RPD_LIMIT}"
            )


class KeyManager:
    def __init__(self, api_keys: List[APIKey]):
        if not api_keys:
            logger.error(
                "No API keys provided to KeyManager. Ensure OR_API_KEYS environment variable is set correctly."
            )
            raise ValueError(
                "No API keys provided to KeyManager. Check the OR_API_KEYS environment variable."
            )

        now = time.time()  # Use Unix timestamp
        # Calculate timestamps for 150 requests evenly distributed over last 24 hours
        day_ago = now - 86400
        time_between_reqs = 86400 / 150  # Time between each request
        historical_times = [day_ago + (i * time_between_reqs) for i in range(150)]

        self.keys: Dict[str, KeyInfo] = {}
        for key in api_keys:
            key_info = KeyInfo(key=key.key, name=key.name)
            key_info.daily_requests = (
                historical_times.copy()
            )  # Each key gets its own copy
            self.keys[key.key] = key_info

        logger.info(f"KeyManager initialized with {len(self.keys)} keys")

    async def get_usable_key(self, request_body: bytes) -> str:
        """
        Finds the least recently used key that is not rate-limited.

        Args:
            request_body: The request body (unused)
        """
        while True:
            # Filter enabled keys and sort by daily usage (fewer used first)
            valid_keys = [k for k in self.keys.values() if k.enabled]
            sorted_keys = sorted(valid_keys, key=lambda k: len(k.daily_requests))

            if not sorted_keys:
                logger.error("No API keys available for the request.")
                raise HTTPException(
                    status_code=503,
                    detail="No API keys available.",
                )

            selected_key_info = None
            for key_info in sorted_keys:
                is_limited = await key_info.is_rate_limited()
                if not is_limited:
                    selected_key_info = key_info
                    break

            if selected_key_info:
                await selected_key_info.record_usage()
                logger.debug(
                    f"Selected key ending in ...{selected_key_info.key[-4:]} (daily: {len(selected_key_info.daily_requests)}/{RPD_LIMIT}, total: {selected_key_info.total_requests})"
                )
                return selected_key_info.key

            # All keys are rate-limited, wait and retry
            logger.warning(
                f"All {len(sorted_keys)} keys are currently rate-limited. Waiting {RETRY_DELAY_SECONDS}s..."
            )
            await asyncio.sleep(RETRY_DELAY_SECONDS)

    def disable_key(self, key: str):
        """Disables a key (e.g., if it's invalid or consistently failing)."""
        if key in self.keys:
            self.keys[key].enabled = False
            logger.warning(f"Disabled key ending in ...{key[-4:]}")

    def enable_key(self, key: str):
        """Re-enables a key."""
        if key in self.keys:
            self.keys[key].enabled = True
            logger.info(f"Enabled key ending in ...{key[-4:]}")


# --- Globals ---
app = FastAPI(title="OpenRouter Reverse Proxy")
key_manager = KeyManager(API_KEYS)  # Initialize key manager with API keys
# Use a single httpx client for connection pooling with proxy and disabled SSL verification
http_client = httpx.AsyncClient(
    timeout=None,  # Global timeout for the client
)


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
                    text = buffer.decode()
                    if any(pattern in text for pattern in RPD_LIMIT_PATTERNS):
                        logger.warning(
                            f"Daily rate limit (RPD) exceeded for key ...{api_key[-4:]}"
                        )
                        key_info = key_manager.keys.get(api_key)
                        if key_info:
                            # Fill daily_requests with 200 timestamps to mark key as rate limited
                            async with key_info.lock:
                                now = time.time()
                                # Clear any old entries first
                                day_ago = now - 86400
                                key_info.daily_requests = [
                                    t for t in key_info.daily_requests if t > day_ago
                                ]
                                # Fill remaining slots up to RPD_LIMIT
                                remaining = RPD_LIMIT - len(key_info.daily_requests)
                                key_info.daily_requests.extend([now] * remaining)
                                logger.warning(
                                    f"Key ...{api_key[-4:]} marked as RPD limited for the next 24 hours"
                                )
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

            # Check for RPD limit in error response
            if any(pattern in error_text for pattern in RPD_LIMIT_PATTERNS):
                logger.warning(
                    f"Daily rate limit (RPD) exceeded for key ...{api_key[-4:]}"
                )
                key_info = key_manager.keys.get(api_key)
                if key_info:
                    await key_info.record_usage(is_completion=True)

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
async def get_stats(request: Request):
    """Returns API key statistics in HTML or plain text format based on Accept header"""
    stats = []
    now = time.time()  # Use Unix timestamp for consistency
    accept = request.headers.get("accept", "")
    want_html = "text/html" in accept

    for key_info in key_manager.keys.values():
        daily_window_start = now - 86400  # 24 hours
        recent_daily_requests = [
            t for t in key_info.daily_requests if t > daily_window_start
        ]
        daily_usage = len(recent_daily_requests)

        # Get all timestamps for requests in the last 24 hours
        recent_timestamps = [
            t for t in key_info.daily_requests if t > daily_window_start
        ]
        recent_timestamps.sort()  # Sort timestamps chronologically

        stats.append(
            {
                "name": key_info.name,
                "key_suffix": key_info.key[-4:],
                "enabled": key_info.enabled,
                "total_requests": key_info.total_requests,
                "daily_usage": daily_usage,
                "last_used": (
                    time.ctime(key_info.last_used_time)
                    if key_info.last_used_time
                    else "Never"
                ),
                "timestamps": recent_timestamps,  # Add timestamps for visualization
            }
        )

    if not want_html:
        # ASCII table format for terminal
        ascii_table = [
            "\n" + "═" * 120,  # Top border
            " API KEY MONITORING DASHBOARD ",
            "═" * 120 + "\n",  # Bottom border of title
        ]

        headers = [
            "NAME",
            "KEY",
            "STATUS",
            "TOTAL",
            "COMPLETIONS",
            "LAST USED",
        ]
        widths = [20, 8, 10, 8, 12, 30]

        # Header
        header_row = " ".join(f"{h:{w}}" for h, w in zip(headers, widths))
        ascii_table.append(header_row)
        ascii_table.append("─" * 120)  # Separator

        # Data rows
        for row in stats:
            status = "✓ ACTIVE" if row["enabled"] else "✗ DISABLED"
            daily = f"{row['daily_usage']}/{RPD_LIMIT}"

            # Color formatting for terminal
            if row["daily_usage"] >= RPD_LIMIT:
                daily = f"\033[1;91m{daily}\033[0m"  # Bold Red
            elif row["daily_usage"] >= RPD_LIMIT * 0.9:  # 90% of limit
                daily = f"\033[1;93m{daily}\033[0m"  # Bold Yellow

            # Construct the row with proper spacing
            formatted_row = [
                f"{row['name'][:19]:<20}",
                f"...{row['key_suffix']:<4} ",
                f"{status:<10}",
                f"{row['total_requests']:<8}",
                f"{daily:<12}",
                f"{row['last_used']:<30}",
            ]
            ascii_table.append(" ".join(formatted_row))

        ascii_table.append("\n" + "═" * 120)  # Bottom border
        ascii_table.append(
            f" Total Requests: {sum(r['total_requests'] for r in stats)}"
        )
        ascii_table.append(
            f" Daily Completions: {sum(r['daily_usage'] for r in stats)}/{RPD_LIMIT}"
        )
        ascii_table.append("═" * 120 + "\n")

        return Response(content="\n".join(ascii_table), media_type="text/plain")

    # Calculate total available completions
    total_available_completions = sum(
        RPD_LIMIT - row["daily_usage"] for row in stats if row["enabled"]
    )

    # HTML format - prepare data for charts including timestamps
    stats_json = json.dumps(
        [
            {
                "name": row["name"],
                "daily_usage": row["daily_usage"],
                "timestamps": row["timestamps"],
            }
            for row in stats
        ]
    )
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "stats_json": stats_json,
            "total_available_completions": total_available_completions,
            "rpd_limit": RPD_LIMIT,  # Pass the RPD_LIMIT constant to the template
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
    Only tracks completion requests against daily limits.
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

    is_completion = "chat/completion" in path
    try:
        selected_key = await key_manager.get_usable_key(body)

        # Only track daily usage for completion requests
        if is_completion and selected_key in key_manager.keys:
            now = time.time()  # Using time.time() consistently
            key_info = key_manager.keys[selected_key]
            async with key_info.lock:
                # Clean up old requests (more than 24h old)
                daily_window_start = now - 86400  # 24 hours
                key_info.daily_requests = [
                    t for t in key_info.daily_requests if t > daily_window_start
                ]
                # Add new request if under limit
                if len(key_info.daily_requests) < RPD_LIMIT:
                    key_info.daily_requests.append(now)
                else:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Daily completion limit reached for key ...{selected_key[-4:]}",
                    )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.critical(f"Failed to get a usable API key: {e}")
        raise HTTPException(
            status_code=503, detail="Service Unavailable: No API keys available."
        )

    target_url = f"{OPENROUTER_API_BASE}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    logger.debug(f"Target URL: {target_url}")

    logger.info(
        f"Forwarding {request.method} request to {target_url} using key ending in ...{selected_key[-4:]}"
    )

    return await _forward_request(request, target_url, selected_key)


# --- Lifecycle Events ---
@app.on_event("startup")
async def app_startup():
    """Start background tasks on startup."""
    logger.info("API proxy server starting up")

    # Log API key configuration on startup
    key_count = len(key_manager.keys)
    if key_count == 0:
        logger.warning(
            "No API keys loaded. Set the OR_API_KEYS environment variable in the format: name1:key1,name2:key2"
        )
    else:
        logger.info(f"Successfully loaded {key_count} API key(s)")


@app.on_event("shutdown")
async def app_shutdown():
    """Close the httpx client gracefully on shutdown."""
    logger.info("Shutting down. Closing HTTP client.")
    await http_client.aclose()


# --- Main Execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting OpenRouter Proxy Server...")

    key_count = len(key_manager.keys)
    if key_count == 0:
        logger.warning("No API keys loaded. Set the OR_API_KEYS environment variable")
    else:
        logger.info(f"Managing {key_count} API key(s)")

    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    workers = int(os.environ.get("WORKERS", "4"))

    # Run in production mode
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,  # Disable reload in production
    )
