# Security-focused Dockerfile for container scanning and hardening
# Multi-stage build with security best practices

# Build stage
FROM python:3.11-slim as builder

# Security: Create non-root user early
RUN groupadd -r testgen && useradd -r -g testgen testgen

# Security: Update system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Security: Set secure working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Security: Install dependencies with verification
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml setup.py ./

# Build the application
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Security: Update system and install security tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -r -g 1001 testgen && \
    useradd -r -u 1001 -g testgen -d /app -s /bin/bash testgen

# Security: Set secure working directory with proper permissions
WORKDIR /app
RUN chown testgen:testgen /app

# Copy built application from builder stage
COPY --from=builder --chown=testgen:testgen /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=testgen:testgen /usr/local/bin /usr/local/bin
COPY --from=builder --chown=testgen:testgen /app/src ./src

# Security: Switch to non-root user
USER testgen

# Security: Set secure environment variables
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Security: Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import testgen_copilot; print('OK')" || exit 1

# Security: Use specific port and expose only what's needed
EXPOSE 8080

# Security: Use explicit command with no shell form
ENTRYPOINT ["python", "-m", "testgen_copilot.cli"]
CMD ["--help"]

# Security: Add labels for metadata
LABEL maintainer="team@testgen.dev" \
      description="TestGen Copilot - Secure container build" \
      version="0.0.1" \
      security.scan="enabled" \
      security.user="non-root" \
      org.opencontainers.image.source="https://github.com/testgen/copilot-assistant" \
      org.opencontainers.image.documentation="https://testgen.readthedocs.io" \
      org.opencontainers.image.licenses="MIT"