FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without installing the project itself)
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project in non-editable mode
RUN uv sync --frozen --no-editable

# Create a non-root user and switch to it
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run the application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
