# TestGen Copilot Assistant - Makefile
# Provides standardized commands for development, testing, and deployment

.PHONY: help install install-dev test test-unit test-integration test-e2e test-security test-coverage lint format type-check security-scan clean build publish docker-build docker-run docs docs-serve pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "TestGen Copilot Assistant - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation and Setup
# =============================================================================

install: ## Install package in production mode
	pip install .

install-dev: ## Install package in development mode with all dependencies
	pip install -e ".[dev,ai]"
	pre-commit install

setup-dev: install-dev ## Complete development environment setup
	@echo "Setting up development environment..."
	@echo "✓ Dependencies installed"
	@echo "✓ Pre-commit hooks installed"
	@echo "✓ Development environment ready!"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linting checks
	ruff check .
	ruff format --check .

format: ## Format code using black and ruff
	black .
	ruff check --fix .
	ruff format .

type-check: ## Run type checking with mypy
	mypy src/testgen_copilot

security-scan: ## Run security scanning with bandit
	bandit -r src/testgen_copilot

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

quality: lint type-check security-scan ## Run all code quality checks

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/ -m "unit" -v

test-integration: ## Run integration tests only
	pytest tests/ -m "integration" -v

test-e2e: ## Run end-to-end tests only
	pytest tests/ -m "e2e" -v

test-security: ## Run security tests only
	pytest tests/ -m "security" -v

test-coverage: ## Run tests with coverage reporting
	pytest --cov=src/testgen_copilot --cov-report=html --cov-report=term-missing

test-performance: ## Run performance tests
	pytest tests/ -m "performance" -v

test-parallel: ## Run tests in parallel
	pytest -n auto

test-watch: ## Run tests in watch mode (requires pytest-watch)
	pytest-watch

# =============================================================================
# Building and Packaging
# =============================================================================

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to test PyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

# =============================================================================
# Docker Operations
# =============================================================================

docker-build: ## Build Docker image
	docker build -t testgen-copilot:latest .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace testgen-copilot:latest

docker-test: ## Run tests in Docker
	docker run --rm -v $(PWD):/workspace testgen-copilot:latest make test

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

# =============================================================================
# Documentation
# =============================================================================

docs-install: ## Install documentation dependencies
	pip install -r docs/requirements.txt

docs-clean: ## Clean documentation build artifacts
	rm -rf docs/_build/
	rm -rf docs/_static/
	mkdir -p docs/_static

docs-apidoc: ## Generate API documentation
	sphinx-apidoc -o docs/api src/testgen_copilot --force --module-first

docs: docs-clean docs-apidoc ## Generate documentation
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation built successfully!"
	@echo "Open docs/_build/html/index.html in your browser"

docs-serve: docs ## Serve documentation locally with auto-reload
	cd docs && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

docs-linkcheck: ## Check documentation for broken links
	cd docs && sphinx-build -b linkcheck . _build/linkcheck

docs-pdf: ## Generate PDF documentation
	cd docs && sphinx-build -b latexpdf . _build/pdf

# =============================================================================
# Development Utilities
# =============================================================================

deps-update: ## Update dependencies
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

deps-sync: ## Sync dependencies
	pip-sync requirements.txt requirements-dev.txt

profile: ## Run performance profiling
	python -m cProfile -o profile.prof -m testgen_copilot.cli --help
	@echo "Profile saved to profile.prof"

benchmark: ## Run benchmarks
	python scripts/benchmark.py

mock-server: ## Start mock API server for testing
	@echo "Mock server not yet implemented"
	@echo "TODO: Add mock LLM API server"

# =============================================================================
# Release Management
# =============================================================================

version-check: ## Check current version
	python -c "from src.testgen_copilot.version import __version__; print(f'Current version: {__version__}')"

changelog: ## Generate changelog
	@echo "Changelog generation not yet implemented"
	@echo "TODO: Add conventional-changelog or similar"

release-dry-run: ## Dry run release process
	semantic-release version --dry-run

release: ## Create new release
	semantic-release version

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci-install: ## Install dependencies for CI
	pip install -e ".[dev]"

ci-test: ## Run tests for CI with coverage
	pytest --cov=src/testgen_copilot --cov-report=xml --cov-fail-under=85

ci-quality: ## Run quality checks for CI
	ruff check .
	bandit -r src/testgen_copilot
	mypy src/testgen_copilot

ci-security: ## Run security checks for CI
	bandit -r src/testgen_copilot -f json -o bandit-report.json
	@echo "Security scan completed"

# =============================================================================
# Environment Management
# =============================================================================

env-create: ## Create virtual environment
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

env-activate: ## Show activation command
	@echo "To activate the environment, run:"
	@echo "source venv/bin/activate"

env-deactivate: ## Show deactivation command
	@echo "To deactivate the environment, run:"
	@echo "deactivate"

# =============================================================================
# Quick Commands
# =============================================================================

dev: install-dev ## Quick development setup
	@echo "Development environment ready!"

check: lint test-unit ## Quick check (lint + unit tests)
	@echo "Quick check completed!"

full-check: quality test ## Full check (all quality + all tests)
	@echo "Full check completed!"

deploy-staging: build ## Deploy to staging (placeholder)
	@echo "Staging deployment not yet implemented"

deploy-prod: build ## Deploy to production (placeholder)
	@echo "Production deployment not yet implemented"