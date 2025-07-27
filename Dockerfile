# Multi-stage build for TestGen Copilot Assistant
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL maintainer="TestGen Team <team@testgen.dev>" \
      org.opencontainers.image.title="TestGen Copilot Assistant" \
      org.opencontainers.image.description="CLI tool for AI-powered test generation and security analysis" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/testgen-team/testgen-copilot-assistant" \
      org.opencontainers.image.url="https://github.com/testgen-team/testgen-copilot-assistant" \
      org.opencontainers.image.documentation="https://testgen-copilot-assistant.readthedocs.io" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# =====================================
# Production stage
# =====================================
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r testgen && useradd -r -g testgen testgen

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy application from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/testgen /usr/local/bin/testgen
COPY --from=builder /build/src /app/src

# Copy additional files
COPY README.md LICENSE ./
COPY docs/ ./docs/
COPY security/ ./security/

# Create directories for application data
RUN mkdir -p /app/data /app/logs /app/cache /app/config && \
    chown -R testgen:testgen /app

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TESTGEN_CACHE_DIR=/app/cache \
    TESTGEN_LOG_FILE=/app/logs/testgen.log \
    TESTGEN_CONFIG_PATH=/app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD testgen --version || exit 1

# Switch to non-root user
USER testgen

# Expose port for potential web interface
EXPOSE 8000

# Default command
ENTRYPOINT ["testgen"]
CMD ["--help"]

# =====================================
# Development stage (for development use)
# =====================================
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install additional development tools
RUN pip install --no-cache-dir \
    ipython \
    jupyterlab \
    pre-commit

# Set development environment variables
ENV TESTGEN_DEV_MODE=true \
    TESTGEN_LOG_LEVEL=DEBUG

# Create development user
RUN useradd -m -s /bin/bash developer
USER developer
WORKDIR /workspace

CMD ["/bin/bash"]

# =====================================
# Testing stage (for CI/CD)
# =====================================
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini .coveragerc ./

# Set testing environment variables
ENV PYTHONPATH=/workspace/src \
    PYTEST_ADDOPTS="--tb=short -v"

# Run tests by default
CMD ["pytest", "tests/", "--cov=src/testgen_copilot", "--cov-report=xml"]