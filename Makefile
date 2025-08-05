# TestGen Copilot Makefile

.PHONY: install test lint format clean build help

help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build package"

install:
	python3 -m pip install --break-system-packages -e ".[dev,ai,security,all]"

test:
	python3 -m pytest tests/ -v --tb=short

lint:
	python3 -m ruff check src/

format:
	python3 -m black src/ tests/
	python3 -m ruff check --fix src/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python3 -m build