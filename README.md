# OpenRouter Proxy

A FastAPI-based reverse proxy for the [OpenRouter API](https://openrouter.ai/docs) that manages multiple API keys and provides key rotation, rate limit tracking, and usage statistics.

## Features

- **API Key Management**: Rotate between multiple OpenRouter API keys to maximize usage
- **Rate Limit Tracking**: Automatically tracks daily rate limits for each key
- **Dashboard**: Visual monitoring of API key usage and availability
- **API Endpoint Compatibility**: Direct drop-in replacement for OpenRouter API endpoints
- **Streaming Support**: Full streaming support for chat completions
- **Redis Persistence**: Optional Redis-based persistence for key states and usage metrics

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer and resolver)
- Docker (optional, for containerized deployment)

## Installation

### Using Docker (Recommended)

```bash
docker pull ghcr.io/theblazehen/openrouter_proxy:latest
```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/theblazehen/openrouter_proxy.git
   cd openrouter_proxy
   ```

2. Install dependencies with uv:
   ```bash
   # Install uv if you don't have it already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies using uv
   uv sync
   ```

## Configuration
The proxy is configured using environment variables. You can set these directly or use a `.env` file (copy from `.env.example`).

| Variable | Description | Format | Default | Required |
|----------|-------------|--------|---------|----------|
| `OR_API_KEYS` | OpenRouter API keys | `name1:key1,name2:key2,...` | None | Yes |
| `OPENROUTER_API_BASE` | OpenRouter API base URL | URL | https://openrouter.ai/api/v1 | No |
| `HOST` | Host to bind the server to | IP address | 0.0.0.0 | No |
| `PORT` | Port to run the server on | Integer | 8000 | No |
| `LOG_LEVEL` | Logging level | debug, info, warning, error, critical | info | No |
| `REDIS_URL` | Redis connection URL for persistence | URL | None | No |

### Example Configuration

```bash
# Direct environment variables
export OR_API_KEYS="personal:sk-or-v1-abc123,work:sk-or-v1-def456"
export HOST="0.0.0.0"
export PORT="8000"
export LOG_LEVEL="info"
```

Or use a `.env` file:

```bash
# Copy the example file
cp .env.example .env
# Edit with your API keys
vim .env
```

## Usage

### Running with Docker

```bash
docker run -p 8000:8000 \
  -e OR_API_KEYS="personal:sk-or-v1-abc123,work:sk-or-v1-def456" \
  ghcr.io/theblazehen/openrouter_proxy:latest
```

### Running manually

```bash
# Activate the project virtual environment
source .venv/bin/activate

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or using uv directly:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. If you're making changes:

```bash
# Update dependencies
uv pip install <new-dependency>

# Lock dependencies
uv pip freeze > requirements.lock

# Sync project (to update after changes)
uv sync
```

### Code Quality Tools

This project uses pre-commit for code quality checks. To set up:

```bash
# Install pre-commit
uv pip install pre-commit

# Install pre-commit hooks
pre-commit install
```

The pre-commit configuration includes:
- Code formatting with Ruff
- Various checks for common issues (trailing whitespace, YAML validity, etc.)
- Code linting with Ruff

## Dashboard
Access the dashboard at `http://localhost:8000/stats` to monitor API key usage and availability.

The dashboard provides:
- Current usage statistics for each API key
- Visual charts showing availability over time
- Predictions for future availability based on usage patterns
- Auto-refreshes every 15 seconds by default

You can also access the dashboard in plain text format by using `curl`:

```bash
curl http://localhost:8000/stats
```

## Security Considerations

When deploying this proxy to production, consider the following:

1. **API Key Protection**: The proxy handles sensitive API keys. Ensure your environment variables are properly secured.
2. **Network Security**: By default, the server listens on all interfaces (0.0.0.0). Consider restricting this in production.
3. **Rate Limiting**: Consider adding rate limiting for the proxy itself to prevent abuse.
4. **Logging**: The proxy logs sensitive information at DEBUG level. Use INFO or higher in production.

## Troubleshooting

### Common Issues

| Issue | Possible Solution |
|-------|-------------------|
| "No API keys available" | Check the OR_API_KEYS environment variable is set correctly |
| Dashboard not updating | Check the browser console for errors, ensure the proxy is running |
| Rate limiting errors | One or more API keys may be at their limits, add more keys |
| High memory usage | Check for memory leaks, reduce historical tracking |
| Redis connection failed | Check Redis URL format and server availability |

### Debugging

Enable DEBUG level logging to see more information:

```bash
export LOG_LEVEL=debug
```

Check the logs for errors:

```bash
tail -f proxy.log
```


## API Endpoints

- **Proxy Endpoint**: `http://localhost:8000/api/v1/{path}`
  - This endpoint proxies all OpenRouter API requests (e.g., `/api/v1/chat/completions`)

- **Stats Dashboard**: `http://localhost:8000/stats`
  - Displays API key usage statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
