# Multi-stage Debian-based Dockerfile for TestGen Copilot
# Optimized for production with security scanning and comprehensive tooling

# ===============================================================================
# Build Stage - Compile dependencies and prepare application
# ===============================================================================
FROM python:3.11-slim-bookworm as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDX_VERSION

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml setup.py requirements.txt requirements-dev.txt ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[dev,ai,security]"

# ===============================================================================
# Production Stage - Minimal runtime environment
# ===============================================================================
FROM python:3.11-slim-bookworm as production

# Set metadata
LABEL org.opencontainers.image.title="TestGen Copilot"
LABEL org.opencontainers.image.description="AI-powered test generation and security analysis"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/testgen-copilot"
LABEL org.opencontainers.image.documentation="https://github.com/terragonlabs/testgen-copilot#readme"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Create non-root user
RUN groupadd -r testgen && useradd -r -g testgen -s /bin/bash testgen

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy application code
COPY --chown=testgen:testgen . .

# Create necessary directories
RUN mkdir -p /workspace/tests /workspace/reports && \
    chown -R testgen:testgen /workspace

# Switch to non-root user
USER testgen

# Set environment variables
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import testgen_copilot; print('OK')" || exit 1

# Default command
ENTRYPOINT ["testgen"]
CMD ["--help"]

# ===============================================================================
# Security Scanner Stage - Extended security tooling
# ===============================================================================
FROM production as scanner

# Switch back to root for tool installation
USER root

# Install additional security tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for vulnerability scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && \
    echo "deb https://aquasecurity.github.io/trivy-repo/deb generic main" | tee -a /etc/apt/sources.list.d/trivy.list && \
    apt-get update && \
    apt-get install -y trivy && \
    rm -rf /var/lib/apt/lists/*

# Install additional security scanning tools
RUN pip install --no-cache-dir \
    semgrep \
    bandit[toml] \
    safety \
    pip-audit \
    cyclonedx-bom

# Create security scan scripts
COPY docker/scripts/security-scan.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/security-scan.sh

# Switch back to testgen user
USER testgen

# Override entrypoint for scanner
ENTRYPOINT ["/usr/local/bin/security-scan.sh"]

# ===============================================================================
# Development Stage - Full development environment
# ===============================================================================
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    less \
    man-db \
    bash-completion \
    tree \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest-watch \
    tox

# Setup bash completion and aliases
RUN echo 'alias ll="ls -la"' >> /home/testgen/.bashrc && \
    echo 'alias la="ls -A"' >> /home/testgen/.bashrc && \
    echo 'alias testgen-dev="testgen --debug"' >> /home/testgen/.bashrc

# Switch back to testgen user
USER testgen

# Override command for development
CMD ["/bin/bash"]