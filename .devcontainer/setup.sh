#!/bin/bash
set -e

echo "🚀 Setting up TestGen Copilot Assistant development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y curl wget git build-essential

# Install Python development dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# Install additional development tools
pip install \
    pre-commit \
    bandit[toml] \
    safety \
    mypy \
    black \
    isort \
    ruff \
    pytest-xdist \
    pytest-benchmark \
    pytest-mock \
    coverage[toml] \
    sphinx \
    sphinx-rtd-theme

# Install Node.js dependencies for potential VS Code extension development
echo "📦 Installing Node.js dependencies..."
npm install -g \
    @vscode/vsce \
    typescript \
    eslint \
    prettier

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {docs/{guides,runbooks},tests/{integration,performance},security/rules,schemas,.github/{workflows,ISSUE_TEMPLATE}}

# Setup git configuration for container
echo "⚙️ Configuring git..."
git config --global --add safe.directory /workspaces/testgen-copilot-assistant
git config --global init.defaultBranch main

# Create initial test configuration
echo "🧪 Setting up test environment..."
cat > pytest.ini << EOF
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src/testgen_copilot
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
EOF

# Setup environment variables
echo "🔐 Setting up environment variables..."
cat > .env.example << EOF
# Development Environment Variables
TESTGEN_LOG_LEVEL=DEBUG
TESTGEN_DEV_MODE=true
TESTGEN_CACHE_DIR=.cache
TESTGEN_CONFIG_PATH=.testgen.config.json

# LLM Configuration (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Testing Configuration
PYTEST_WORKERS=auto
COVERAGE_TARGET=80
QUALITY_TARGET=85

# Security Configuration
BANDIT_CONFIG_FILE=.bandit
SAFETY_DB_PATH=.safety_db

# Performance Configuration
BENCHMARK_SAVE_DATA=true
BENCHMARK_COMPARE=true
EOF

# Create initial project metrics file
echo "📊 Creating project metrics..."
mkdir -p .github
cat > .github/project-metrics.json << EOF
{
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sdlc_completeness": 25,
  "automation_coverage": 40,
  "security_score": 60,
  "documentation_health": 70,
  "test_coverage": 75,
  "deployment_reliability": 50,
  "maintenance_automation": 30,
  "metrics_version": "1.0.0"
}
EOF

echo "✅ Development environment setup complete!"
echo "🎯 Next steps:"
echo "   1. Copy .env.example to .env and configure your settings"
echo "   2. Run 'pytest' to execute the test suite"
echo "   3. Run 'pre-commit run --all-files' to check code quality"
echo "   4. Start developing with 'python -m testgen_copilot --help'"