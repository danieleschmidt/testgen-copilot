# TestGen Copilot - Enhanced Justfile
# Modern task runner with advanced automation capabilities
# See https://github.com/casey/just for documentation

# Configuration
set shell := ["bash", "-uc"]
set dotenv-load := true

# Variables
python := env_var_or_default("PYTHON", "python")
venv_path := env_var_or_default("VIRTUAL_ENV", ".venv")
test_args := env_var_or_default("TEST_ARGS", "")
parallel_jobs := env_var_or_default("PARALLEL_JOBS", "auto")

# Colors for output
export RED := '\033[0;31m'
export GREEN := '\033[0;32m'
export YELLOW := '\033[1;33m'
export BLUE := '\033[0;34m'
export NC := '\033[0m'

# Show available recipes with enhanced formatting
default:
    @echo -e "${BLUE}🚀 TestGen Copilot - Available Commands${NC}"
    @echo ""
    @just --list --unsorted | grep -E '^[[:space:]]*[[:alnum:]_-]+' | sort

# ===============================================================================
# Environment Setup
# ===============================================================================

# Complete development environment setup
setup: install-system-deps create-venv install-dev setup-git-hooks setup-vscode
    @echo -e "${GREEN}✅ Development environment setup complete!${NC}"

# Create virtual environment
create-venv:
    @echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    {{python}} -m venv {{venv_path}}
    @echo -e "${GREEN}✅ Virtual environment created at {{venv_path}}${NC}"

# Install system dependencies (platform-aware)
install-system-deps:
    #!/usr/bin/env bash
    echo -e "${BLUE}🔧 Installing system dependencies...${NC}"
    
    if command -v apt-get >/dev/null 2>&1; then
        # Ubuntu/Debian
        sudo apt-get update && sudo apt-get install -y git curl jq
    elif command -v yum >/dev/null 2>&1; then
        # RHEL/CentOS
        sudo yum install -y git curl jq
    elif command -v brew >/dev/null 2>&1; then
        # macOS
        brew install git curl jq
    elif command -v pacman >/dev/null 2>&1; then
        # Arch Linux
        sudo pacman -S --noconfirm git curl jq
    else
        echo -e "${YELLOW}⚠️  Unknown package manager. Please install git, curl, and jq manually.${NC}"
    fi

# Install development dependencies
install-dev:
    @echo -e "${BLUE}📦 Installing development dependencies...${NC}"
    {{python}} -m pip install --upgrade pip setuptools wheel
    {{python}} -m pip install -e ".[dev,ai,security,all]"
    @echo -e "${GREEN}✅ Development dependencies installed${NC}"

# Install production dependencies only
install:
    @echo -e "${BLUE}📦 Installing production dependencies...${NC}"
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -e ".[ai]"
    @echo -e "${GREEN}✅ Production dependencies installed${NC}"

# Setup git hooks
setup-git-hooks:
    @echo -e "${BLUE}🎣 Setting up git hooks...${NC}"
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg
    @echo -e "${GREEN}✅ Git hooks configured${NC}"

# Setup VS Code workspace
setup-vscode:
    @echo -e "${BLUE}🛠️  Setting up VS Code workspace...${NC}"
    mkdir -p .vscode
    @echo -e "${GREEN}✅ VS Code workspace configured${NC}"

# ===============================================================================
# Code Quality and Testing
# ===============================================================================

# Run all tests with coverage
test *args=test_args:
    @echo -e "${BLUE}🧪 Running tests...${NC}"
    pytest {{args}} --cov=src/testgen_copilot --cov-report=term-missing --cov-report=html

# Run tests in parallel
test-parallel *args="":
    @echo -e "${BLUE}⚡ Running tests in parallel...${NC}"
    pytest -n {{parallel_jobs}} {{args}}

# Run specific test types
test-unit:
    @echo -e "${BLUE}🔬 Running unit tests...${NC}"
    pytest tests/ -m "unit" -v

test-integration:
    @echo -e "${BLUE}🔗 Running integration tests...${NC}"
    pytest tests/ -m "integration" -v

test-e2e:
    @echo -e "${BLUE}🌐 Running end-to-end tests...${NC}"
    pytest tests/ -m "e2e" -v

test-security:
    @echo -e "${BLUE}🛡️  Running security tests...${NC}"
    pytest tests/ -m "security" -v

test-performance:
    @echo -e "${BLUE}⚡ Running performance tests...${NC}"
    pytest tests/ -m "performance" -v --benchmark-only

# Run mutation testing
test-mutation:
    @echo -e "${BLUE}🧬 Running mutation testing...${NC}"
    mutmut run --paths-to-mutate src/testgen_copilot/

# Continuous testing (watch mode)
test-watch:
    @echo -e "${BLUE}👀 Starting continuous testing...${NC}"
    ptw -- --testmon

# Format code with multiple tools
format:
    @echo -e "${BLUE}💅 Formatting code...${NC}"
    black .
    ruff check --fix .
    ruff format .
    @echo -e "${GREEN}✅ Code formatted${NC}"

# Check formatting without making changes
format-check:
    @echo -e "${BLUE}🔍 Checking code formatting...${NC}"
    black --check .
    ruff format --check .

# Run all linting checks
lint:
    @echo -e "${BLUE}🔍 Running linting checks...${NC}"
    ruff check .
    @echo -e "${GREEN}✅ Linting complete${NC}"

# Type checking
typecheck:
    @echo -e "${BLUE}🔍 Running type checks...${NC}"
    mypy src/testgen_copilot
    @echo -e "${GREEN}✅ Type checking complete${NC}"

# Security scanning
security:
    @echo -e "${BLUE}🛡️  Running security scans...${NC}"
    bandit -r src/testgen_copilot
    safety check
    pip-audit --require-hashes --desc
    @echo -e "${GREEN}✅ Security scanning complete${NC}"

# Comprehensive quality check
check: lint typecheck security test-unit
    @echo -e "${GREEN}✅ All quality checks passed${NC}"

# Run pre-commit on all files
precommit:
    @echo -e "${BLUE}🎣 Running pre-commit hooks...${NC}"
    pre-commit run --all-files

# ===============================================================================
# Documentation
# ===============================================================================

# Install documentation dependencies
docs-deps:
    @echo -e "${BLUE}📚 Installing documentation dependencies...${NC}"
    {{python}} -m pip install -r docs/requirements.txt

# Generate API documentation
docs-api:
    @echo -e "${BLUE}📖 Generating API documentation...${NC}"
    sphinx-apidoc -o docs/api src/testgen_copilot --force --module-first

# Build documentation
docs: docs-deps docs-api
    @echo -e "${BLUE}📚 Building documentation...${NC}"
    cd docs && sphinx-build -b html . _build/html
    @echo -e "${GREEN}✅ Documentation built at docs/_build/html/index.html${NC}"

# Serve documentation locally
docs-serve: docs
    @echo -e "${BLUE}🌐 Serving documentation at http://localhost:8000${NC}"
    cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

# Check documentation links
docs-linkcheck:
    @echo -e "${BLUE}🔗 Checking documentation links...${NC}"
    cd docs && sphinx-build -b linkcheck . _build/linkcheck

# Generate changelog
changelog *args="":
    @echo -e "${BLUE}📝 Generating changelog...${NC}"
    {{python}} scripts/changelog-generator.py {{args}}

# ===============================================================================
# Build and Release
# ===============================================================================

# Clean all build artifacts
clean:
    @echo -e "${BLUE}🧹 Cleaning build artifacts...${NC}"
    rm -rf build/ dist/ *.egg-info/
    rm -rf .pytest_cache/ .coverage htmlcov/
    rm -rf .ruff_cache/ .mypy_cache/
    rm -rf docs/_build/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    @echo -e "${GREEN}✅ Cleanup complete${NC}"

# Build distribution packages
build: clean
    @echo -e "${BLUE}📦 Building distribution packages...${NC}"
    {{python}} -m build
    @echo -e "${GREEN}✅ Build complete${NC}"

# Build VS Code extension
build-vscode:
    @echo -e "${BLUE}🔧 Building VS Code extension...${NC}"
    cd vscode-extension && npm install && npm run compile && npm run package
    @echo -e "${GREEN}✅ VS Code extension built${NC}"

# Validate package
validate: build
    @echo -e "${BLUE}✅ Validating package...${NC}"
    twine check dist/*
    @echo -e "${GREEN}✅ Package validation complete${NC}"

# Publish to test PyPI
publish-test: validate
    @echo -e "${BLUE}🚀 Publishing to test PyPI...${NC}"
    twine upload --repository testpypi dist/*

# Publish to PyPI
publish: validate
    @echo -e "${BLUE}🚀 Publishing to PyPI...${NC}"
    twine upload dist/*

# Create release (with semantic versioning)
release *args="":
    @echo -e "${BLUE}🏷️  Creating release...${NC}"
    semantic-release version {{args}}

# ===============================================================================
# Container Operations
# ===============================================================================

# Build Docker image
docker-build tag="latest":
    @echo -e "${BLUE}🐳 Building Docker image...${NC}"
    docker build -t testgen-copilot:{{tag}} .

# Build multi-platform Docker images
docker-build-multi tag="latest":
    @echo -e "${BLUE}🐳 Building multi-platform Docker images...${NC}"
    ./scripts/container-automation.sh build

# Run container security scan
docker-scan tag="latest":
    @echo -e "${BLUE}🛡️  Scanning Docker image for vulnerabilities...${NC}"
    ./scripts/container-automation.sh scan

# Run Docker container
docker-run tag="latest" *args="":
    @echo -e "${BLUE}🐳 Running Docker container...${NC}"
    docker run -it --rm -v $(PWD):/workspace testgen-copilot:{{tag}} {{args}}

# Start development services
services-up:
    @echo -e "${BLUE}🚀 Starting development services...${NC}"
    docker-compose -f monitoring/docker-compose.observability.yml up -d

# Stop development services
services-down:
    @echo -e "${BLUE}🛑 Stopping development services...${NC}"
    docker-compose -f monitoring/docker-compose.observability.yml down

# ===============================================================================
# Development Utilities
# ===============================================================================

# Profile application performance
profile *args="--help":
    @echo -e "${BLUE}📊 Profiling application...${NC}"
    {{python}} -m cProfile -o profile.prof -m testgen_copilot.cli {{args}}
    @echo -e "${GREEN}✅ Profile saved to profile.prof${NC}"

# Run load testing
load-test:
    @echo -e "${BLUE}⚡ Running load tests...${NC}"
    locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 30s

# Generate test data
generate-test-data:
    @echo -e "${BLUE}🎲 Generating test data...${NC}"
    {{python}} scripts/generate_tests.py

# Update dependencies
deps-update:
    @echo -e "${BLUE}🔄 Updating dependencies...${NC}"
    {{python}} -m pip install --upgrade pip-tools
    pip-compile --upgrade requirements.in
    pip-compile --upgrade requirements-dev.in

# Sync dependencies
deps-sync:
    @echo -e "${BLUE}🔄 Syncing dependencies...${NC}"
    pip-sync requirements.txt requirements-dev.txt

# Check for outdated packages
deps-outdated:
    @echo -e "${BLUE}📦 Checking for outdated packages...${NC}"
    {{python}} -m pip list --outdated

# Run autonomous execution in dry-run mode
autonomous-dry-run:
    @echo -e "${BLUE}🤖 Running autonomous execution (dry-run)...${NC}"
    testgen-autonomous --dry-run --verbose

# Generate SBOM (Software Bill of Materials)
sbom:
    @echo -e "${BLUE}📋 Generating SBOM...${NC}"
    cyclonedx-py --output-format json --output-file sbom.json .

# ===============================================================================
# Monitoring and Observability
# ===============================================================================

# View application logs
logs:
    @echo -e "${BLUE}📄 Viewing application logs...${NC}"
    tail -f logs/testgen.log 2>/dev/null || echo "No log file found"

# Monitor system resources
monitor:
    @echo -e "${BLUE}📊 Monitoring system resources...${NC}"
    htop || top

# Health check
health:
    @echo -e "${BLUE}💓 Running health check...${NC}"
    {{python}} -c "import testgen_copilot; print('✅ TestGen Copilot is healthy')"

# ===============================================================================
# Git and Version Control
# ===============================================================================

# Initialize git repository with best practices
git-init:
    @echo -e "${BLUE}📝 Initializing git repository...${NC}"
    git init
    git add .
    git commit -m "feat: initial commit\n\n🚀 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

# Clean git repository
git-clean:
    @echo -e "${BLUE}🧹 Cleaning git repository...${NC}"
    git clean -fd
    git reset --hard HEAD

# Show git status with enhanced formatting
status:
    @echo -e "${BLUE}📊 Git Status${NC}"
    git status --short --branch

# ===============================================================================
# Help and Information
# ===============================================================================

# Show system information
info:
    @echo -e "${BLUE}ℹ️  System Information${NC}"
    @echo "Python: $({{python}} --version)"
    @echo "Platform: $(uname -s) $(uname -r)"
    @echo "Architecture: $(uname -m)"
    @echo "Working Directory: $(pwd)"
    @echo "Virtual Environment: {{venv_path}}"

# Validate development environment
validate-env:
    @echo -e "${BLUE}✅ Validating development environment...${NC}"
    {{python}} --version
    pip --version
    git --version
    @echo -e "${GREEN}✅ Development environment is valid${NC}"

# Show available recipes in specific categories
help-dev:
    @echo -e "${BLUE}🛠️  Development Commands${NC}"
    @just --list | grep -E "(setup|install|test|format|lint|check)"

help-build:
    @echo -e "${BLUE}📦 Build Commands${NC}"
    @just --list | grep -E "(build|publish|release|clean)"

help-docker:
    @echo -e "${BLUE}🐳 Docker Commands${NC}"
    @just --list | grep -E "docker"

help-docs:
    @echo -e "${BLUE}📚 Documentation Commands${NC}"
    @just --list | grep -E "docs"