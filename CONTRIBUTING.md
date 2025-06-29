# Contributing to TestGen Copilot Assistant

Thank you for your interest in improving this project!

## Development Setup
1. Install Python 3.11.
2. Run `pip install -e .`
3. Install dev tools: `pip install ruff bandit pytest coverage`

## Checks Before Commit
```bash
ruff check .
bandit -r src
pytest --cov=src --cov-report=term
```
The CI workflow runs these same steps.

## Pull Request Guidelines
- Keep changes focused and well described.
- Ensure all tests and linters pass.
- Provide unit tests for new functionality.
