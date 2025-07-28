# TestGen Copilot - Justfile
# Alternative task runner - see https://github.com/casey/just

# Show available recipes
default:
    @just --list

# Install development dependencies
install:
    pip install -e ".[dev,ai]"
    pre-commit install

# Run tests
test:
    pytest

# Format and lint code
format:
    black .
    ruff check --fix .
    ruff format .

# Run all quality checks
check:
    ruff check .
    mypy src/testgen_copilot
    bandit -r src/testgen_copilot

# Build package
build:
    python -m build

# Clean build artifacts
clean:
    rm -rf build/ dist/ *.egg-info/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -exec rm -rf {} +