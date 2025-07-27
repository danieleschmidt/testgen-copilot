# TestGen Copilot Assistant - Makefile
# 
# Common development tasks and build automation

.PHONY: help install install-dev clean lint format type-check test test-unit test-integration test-performance
.PHONY: test-security test-coverage build build-dev docker docker-dev docker-test docker-clean
.PHONY: docs docs-serve release release-test pre-commit security-scan benchmark
.PHONY: setup-dev setup-hooks validate quality-check ci-check

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
PYTEST := pytest
DOCKER := docker
COMPOSE := docker-compose

# Get version from pyproject.toml
VERSION := $(shell grep '^version = ' pyproject.toml | cut -d '"' -f 2)

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)TestGen Copilot Assistant - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Setup/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && !/Setup/ && !/Docker/ && !/Release/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Docker/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Release Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Release/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup commands
setup-dev: ## Setup - Install development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Development environment setup complete!$(NC)"

setup-hooks: ## Setup - Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

install: ## Install package for production
	@echo "$(BLUE)Installing TestGen Copilot...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: setup-dev setup-hooks ## Install package with development dependencies

# Code quality commands
clean: ## Remove build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf logs/
	rm -rf profiles/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Clean complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatting complete!$(NC)"

lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code...$(NC)"
	ruff check src/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/testgen_copilot/
	@echo "$(GREEN)Type checking complete!$(NC)"

security-scan: ## Run security scanning with bandit
	@echo "$(BLUE)Running security scan...$(NC)"
	bandit -r src/ -f json -o security-report.json || true
	bandit -r src/
	@echo "$(GREEN)Security scan complete!$(NC)"

# Testing commands
test: test-unit ## Run all tests
	@echo "$(GREEN)All tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/ -m "unit or not (integration or performance or e2e)" -v

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/ -m "integration" -v

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTEST) tests/ -m "performance" -v

test-security: ## Run security tests
	@echo "$(BLUE)Running security tests...$(NC)"
	$(PYTEST) tests/ -m "security" -v

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(PYTEST) tests/ -m "e2e" -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=src/testgen_copilot --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

benchmark: ## Run benchmark tests
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTEST) tests/ -m "benchmark" --benchmark-only --benchmark-save=latest
	@echo "$(GREEN)Benchmark complete!$(NC)"

# Build commands
build: clean ## Build package for distribution
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Build complete! Check dist/ directory$(NC)"

build-dev: ## Build development package
	@echo "$(BLUE)Building development package...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)Development build complete!$(NC)"

# Docker commands
docker: ## Docker - Build production image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t testgen-copilot:$(VERSION) .
	$(DOCKER) tag testgen-copilot:$(VERSION) testgen-copilot:latest
	@echo "$(GREEN)Docker image built: testgen-copilot:$(VERSION)$(NC)"

docker-dev: ## Docker - Build and run development container
	@echo "$(BLUE)Building and running development container...$(NC)"
	$(COMPOSE) build testgen-dev
	$(COMPOSE) run --rm testgen-dev

docker-test: ## Docker - Run tests in container
	@echo "$(BLUE)Running tests in Docker container...$(NC)"
	$(COMPOSE) build testgen-test
	$(COMPOSE) run --rm testgen-test

docker-clean: ## Docker - Clean up containers and images
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(COMPOSE) down -v
	$(DOCKER) system prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

# Documentation commands
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)Documentation generated in docs/_build/html/$(NC)"

docs-serve: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation on http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Quality assurance
pre-commit: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks complete!$(NC)"

validate: lint type-check test-unit ## Run basic validation (lint, type-check, unit tests)
	@echo "$(GREEN)Validation complete!$(NC)"

quality-check: format lint type-check security-scan test-coverage ## Run comprehensive quality checks
	@echo "$(GREEN)Quality check complete!$(NC)"

ci-check: lint type-check test security-scan ## Run CI pipeline checks
	@echo "$(GREEN)CI checks complete!$(NC)"

# Release commands
release-test: build ## Release - Test package on TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	twine check dist/*
	twine upload --repository testpypi dist/*
	@echo "$(GREEN)Test release complete!$(NC)"

release: build ## Release - Upload package to PyPI
	@echo "$(BLUE)Uploading to PyPI...$(NC)"
	twine check dist/*
	twine upload dist/*
	@echo "$(GREEN)Release complete!$(NC)"

# Utility commands
version: ## Show current version
	@echo "$(BLUE)Current version: $(VERSION)$(NC)"

env-check: ## Check development environment
	@echo "$(BLUE)Checking development environment...$(NC)"
	@$(PYTHON) --version
	@$(PIP) --version
	@echo "TestGen version: $(VERSION)"
	@echo "$(GREEN)Environment check complete!$(NC)"

# Database commands (for future use)
db-setup: ## Setup database (placeholder for future)
	@echo "$(YELLOW)Database setup not yet implemented$(NC)"

db-migrate: ## Run database migrations (placeholder for future)
	@echo "$(YELLOW)Database migrations not yet implemented$(NC)"

# Monitoring setup (for future use)
monitoring-up: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	$(COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)Monitoring stack started!$(NC)"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

monitoring-down: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	$(COMPOSE) --profile monitoring down
	@echo "$(GREEN)Monitoring stack stopped!$(NC)"

# Performance profiling
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profile...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m testgen_copilot --help
	@echo "$(GREEN)Profile saved to profile.stats$(NC)"

# Generate project metrics
metrics: ## Generate project metrics
	@echo "$(BLUE)Generating project metrics...$(NC)"
	@echo "Lines of code:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test files:"
	@find tests/ -name "test_*.py" | wc -l
	@echo "$(GREEN)Metrics generation complete!$(NC)"