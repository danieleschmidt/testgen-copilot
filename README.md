# testgen-copilot

Python AST-based unit test stub generator and vulnerability scanner. No LLM needed — pure static analysis.

## Features

- **Test stub generation**: Analyzes functions, generates pytest stubs with edge cases (None, empty, type errors)
- **Vulnerability scanning**: Detects SQL injection, hardcoded secrets, eval usage, shell injection, pickle usage
- **Untested function detection**: Compares source vs test file to find gaps

## Usage

```bash
# Generate test stubs
python -m testgen.cli gen mymodule.py > tests/test_mymodule.py

# Scan for vulnerabilities
python -m testgen.cli scan mymodule.py

# Find untested functions
python -m testgen.cli untested mymodule.py --tests tests/test_mymodule.py
```

## Install

```bash
pip install -e .
# then use the CLI:
testgen gen mymodule.py
testgen scan mymodule.py
testgen untested mymodule.py --tests tests/test_mymodule.py
```

## Testing

```bash
pytest tests/ -v
```
