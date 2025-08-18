# TestGen Copilot Makefile
# Production-ready build automation with comprehensive quality gates

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := testgen-copilot
VERSION := $(shell $(PYTHON) -c "import src.testgen_copilot.version; print(src.testgen_copilot.version.__version__)")
DOCKER_REGISTRY := docker.io
DOCKER_NAMESPACE := terragonlabs
DOCKER_IMAGE := $(DOCKER_REGISTRY)/$(DOCKER_NAMESPACE)/$(PROJECT_NAME)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help install install-dev test test-all test-coverage test-security \
        lint format type-check security-scan clean build build-docker \
        publish release deploy docs serve-docs pre-commit setup-hooks \
        benchmark profile validate-dependencies check-vulnerabilities

help: ## Show this help message
	@echo "$(BLUE)TestGen Copilot Build System$(NC)"
	@echo "$(BLUE)=============================$(NC)"
	@echo ""
	@echo "$(GREEN)Installation targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ && /install/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ && /test|lint|format|type/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Build and Deploy targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ && /build|deploy|publish|release/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Other targets:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ && !/install|test|lint|format|type|build|deploy|publish|release/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,ai,security,api,monitoring,database,all]"
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing targets
test: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

test-all: ## Run all tests including slow ones
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ --cov=src/testgen_copilot --cov-report=html --cov-report=xml --cov-report=term-missing --cov-fail-under=80

test-security: ## Run security-focused tests
	@echo "$(BLUE)Running security tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m security

test-performance: ## Run performance benchmarks
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m performance --benchmark-only

# Code quality targets
lint: ## Run code linting
	@echo "$(BLUE)Running linting...$(NC)"
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m black --check src/ tests/

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/
	$(PYTHON) -m isort src/ tests/

type-check: ## Run static type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	$(PYTHON) -m mypy src/testgen_copilot

# Security targets
security-scan: ## Run security vulnerability scanning
	@echo "$(BLUE)Running security scans...$(NC)"
	$(PYTHON) -m bandit -r src/ -f json -o security-report.json || true
	$(PYTHON) -m safety check --json || true
	@echo "$(GREEN)Security scan complete. Check security-report.json$(NC)"

check-vulnerabilities: ## Check for known vulnerabilities
	@echo "$(BLUE)Checking for vulnerabilities...$(NC)"
	$(PYTHON) -m safety check --full-report

validate-dependencies: ## Validate all dependencies
	@echo "$(BLUE)Validating dependencies...$(NC)"
	$(PIP) check
	$(PYTHON) -m pip-audit

# Pre-commit targets
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

setup-hooks: ## Set up git hooks
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg

# Build targets
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	find . -type f -name "*.rej" -delete

build: clean validate-dependencies ## Build package distributions
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete. Check dist/ directory.$(NC)"

build-docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):latest -t $(DOCKER_IMAGE):$(VERSION) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(VERSION)$(NC)"

build-docker-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE):dev .

# Quality gates
quality-gate: lint type-check test-coverage security-scan ## Run all quality gates
	@echo "$(GREEN)All quality gates passed!$(NC)"

# Documentation targets
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && $(PYTHON) -m sphinx -b html . _build/html

serve-docs: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# Performance targets
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) -m pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m testgen_copilot --help
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Release targets
publish-test: build ## Publish to PyPI test
	@echo "$(BLUE)Publishing to PyPI test...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	@echo "$(BLUE)Publishing to PyPI...$(NC)"
	$(PYTHON) -m twine upload dist/*

release: quality-gate build ## Create a new release
	@echo "$(BLUE)Creating release $(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	@echo "$(GREEN)Release $(VERSION) created!$(NC)"

# Deployment targets
deploy-dev: build-docker ## Deploy to development environment
	@echo "$(BLUE)Deploying to development...$(NC)"
	docker-compose -f docker-compose.yml up -d

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	docker-compose -f docker-compose.prod.yml up -d

# Utility targets
version: ## Show current version
	@echo "$(GREEN)Current version: $(VERSION)$(NC)"

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Git branch: $(shell git rev-parse --abbrev-ref HEAD)"
	@echo "Git commit: $(shell git rev-parse --short HEAD)"
	@echo "Docker image: $(DOCKER_IMAGE):$(VERSION)"

logs: ## Show application logs
	docker-compose logs -f testgen-copilot

# CI/CD helpers
ci-install: ## Install dependencies for CI
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,ai,security]"

ci-test: ## Run tests in CI environment
	$(PYTHON) -m pytest tests/ --cov=src/testgen_copilot --cov-report=xml --junitxml=junit.xml

ci-security: ## Run security checks in CI
	$(PYTHON) -m bandit -r src/ -f json -o bandit-report.json
	$(PYTHON) -m safety check --json --output safety-report.json

# Development helpers
dev-setup: install-dev setup-hooks ## Complete development environment setup
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-reset: clean install-dev ## Reset development environment
	@echo "$(GREEN)Development environment reset!$(NC)"

# Quick commands for common workflows
quick-test: format lint test ## Quick development test cycle
	@echo "$(GREEN)Quick test cycle complete!$(NC)"

quick-build: clean build build-docker ## Quick build cycle
	@echo "$(GREEN)Quick build cycle complete!$(NC)"

# Database operations (for quantum planner)
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -m testgen_copilot.database.migrations init

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(PYTHON) -m testgen_copilot.database.migrations migrate

db-reset: ## Reset database
	@echo "$(BLUE)Resetting database...$(NC)"
	$(PYTHON) -m testgen_copilot.database.migrations reset