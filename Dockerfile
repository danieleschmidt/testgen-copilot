# Multi-stage Dockerfile for TestGen Copilot Assistant
# Optimized for security, size, and performance

# =============================================================================
# Build Stage - Compile dependencies and prepare application
# =============================================================================
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r testgen && useradd -r -g testgen testgen

# Set work directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml setup.py README.md ./
COPY src/ src/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install build && \
    python -m build && \
    pip wheel --wheel-dir /wheels dist/*.whl

# =============================================================================
# Runtime Stage - Minimal production image
# =============================================================================
FROM python:3.11-slim-bullseye as runtime

# Set build metadata
LABEL org.opencontainers.image.title="TestGen Copilot Assistant" \
      org.opencontainers.image.description="AI-powered test generation and security analysis CLI tool" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="TestGen Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/testgen/copilot-assistant"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TESTGEN_ENV=production \
    TESTGEN_LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r testgen && useradd -r -g testgen -d /home/testgen testgen

# Create necessary directories
RUN mkdir -p /app /home/testgen/.testgen && \
    chown -R testgen:testgen /app /home/testgen

# Set work directory
WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /wheels /wheels

# Install the application
RUN pip install --upgrade pip && \
    pip install --find-links /wheels testgen_copilot && \
    rm -rf /wheels

# Copy application files
COPY --chown=testgen:testgen . .

# Switch to non-root user
USER testgen

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD testgen --version || exit 1

# Expose any necessary ports (if running as server in future)
# EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["testgen"]

# Default command
CMD ["--help"]

# =============================================================================
# Development Stage - For development with all tools
# =============================================================================
FROM runtime as development

# Switch back to root for installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY --from=builder /app/pyproject.toml ./
RUN pip install -e ".[dev,ai]"

# Install additional development tools
RUN pip install \
    pytest-watch \
    jupyterlab \
    ipython

# Switch back to testgen user
USER testgen

# Set development environment
ENV TESTGEN_ENV=development \
    TESTGEN_LOG_LEVEL=DEBUG

# Default command for development
CMD ["bash"]

# =============================================================================
# Testing Stage - For running tests in CI/CD
# =============================================================================
FROM development as testing

# Copy test files
COPY --chown=testgen:testgen tests/ tests/

# Run tests by default
CMD ["pytest", "--cov=src/testgen_copilot", "--cov-report=xml", "--cov-report=term-missing"]