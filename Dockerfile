FROM python:3.12-slim-bookworm

# Install uv by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Ensure the virtual environment is created in the project directory
ENV UV_LINK_MODE=copy

# Copy project files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN /uv/bin/uv sync --frozen --no-dev

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8500

# Use 'uv run' to execute the application within the managed environment
# This avoids issues with PATH or missing modules in the system python
CMD ["/uv/bin/uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8500"]