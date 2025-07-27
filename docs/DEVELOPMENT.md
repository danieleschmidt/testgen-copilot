# Development Guide

This guide provides comprehensive information for developers contributing to TestGen Copilot Assistant.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)
- VS Code (recommended) or your preferred IDE

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/testgen-team/testgen-copilot-assistant.git
   cd testgen-copilot-assistant
   ```

2. **Set up development environment**
   ```bash
   make install-dev
   ```

3. **Verify installation**
   ```bash
   testgen --version
   make test
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make quality-check
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a pull request**
   - Use our PR template
   - Ensure all CI checks pass
   - Request review from maintainers

## Development Environment

### Local Development

#### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run hooks manually
pre-commit run --all-files
```

### Docker Development

#### Using Docker Compose
```bash
# Start development environment
docker-compose up testgen-dev

# Run tests in container
docker-compose run --rm testgen-test

# Build production image
docker-compose build testgen
```

#### Development Container
```bash
# Build development image
docker build --target development -t testgen-dev .

# Run interactive development container
docker run -it --rm -v $(pwd):/workspace testgen-dev
```

### VS Code Setup

#### Recommended Extensions
- Python
- Pylint
- Black Formatter
- isort
- MyPy Type Checker
- GitHub Copilot
- GitLens
- Docker

#### Configuration
The repository includes VS Code configuration in `.vscode/`:
- `settings.json`: Editor and extension settings
- `tasks.json`: Common development tasks
- `launch.json`: Debug configurations

## Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Import Ordering**: isort with Black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public functions and classes

#### Formatting Tools
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

#### Linting
```bash
# Run linter
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

#### Type Checking
```bash
# Run type checker
mypy src/testgen_copilot/
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

#### Examples
```
feat(cli): add support for Go language analysis
fix(security): resolve SQL injection detection false positive
docs(api): update REST API documentation
test(integration): add end-to-end CLI tests
```

### Code Architecture

#### Project Structure
```
src/testgen_copilot/
├── __init__.py          # Package initialization
├── __main__.py          # CLI entry point
├── cli.py               # Command-line interface
├── core.py              # Main application logic
├── ast_utils.py         # AST parsing utilities
├── generator.py         # Test generation engine
├── security.py          # Security analysis
├── coverage.py          # Coverage analysis
├── quality.py           # Quality assessment
├── file_utils.py        # File operations
├── cache.py             # Caching layer
├── logging_config.py    # Logging configuration
└── ...
```

#### Design Principles
1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Use dependency injection for testability
3. **Configuration-Driven**: Behavior controlled by configuration
4. **Extensibility**: Plugin architecture for new languages/frameworks
5. **Error Handling**: Comprehensive error handling and logging

### API Design

#### Function Signatures
```python
def generate_tests(
    source_file: Path,
    output_dir: Path,
    config: Optional[Config] = None,
    *,
    coverage_target: float = 0.8,
    include_edge_cases: bool = True
) -> List[TestCase]:
    """Generate test cases for the given source file.
    
    Args:
        source_file: Path to the source code file
        output_dir: Directory to write generated tests
        config: Optional configuration object
        coverage_target: Target code coverage percentage
        include_edge_cases: Whether to generate edge case tests
        
    Returns:
        List of generated test cases
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        ValueError: If coverage_target is not between 0 and 1
    """
```

#### Error Handling
```python
from testgen_copilot.exceptions import TestGenError, ConfigurationError

class SecurityAnalysisError(TestGenError):
    """Raised when security analysis fails."""
    pass

def analyze_security(code: str) -> SecurityReport:
    try:
        # Analysis logic
        pass
    except Exception as e:
        raise SecurityAnalysisError(f"Security analysis failed: {e}") from e
```

## Testing

### Test Structure

```
tests/
├── unit/                # Unit tests
│   ├── test_cli.py
│   ├── test_generator.py
│   └── ...
├── integration/         # Integration tests
│   ├── test_cli_integration.py
│   └── ...
├── performance/         # Performance tests
│   ├── test_benchmark_core.py
│   └── ...
├── e2e/                # End-to-end tests
│   └── ...
├── conftest.py         # Pytest configuration
└── fixtures/           # Test fixtures
```

### Running Tests

#### All Tests
```bash
make test
```

#### Specific Test Categories
```bash
# Unit tests only
pytest tests/ -m "unit"

# Integration tests
pytest tests/ -m "integration"

# Performance tests
pytest tests/ -m "performance"

# With coverage
pytest tests/ --cov=src/testgen_copilot --cov-report=html
```

#### Test Markers
```python
@pytest.mark.unit
def test_parse_python_file():
    """Unit test for Python file parsing."""
    pass

@pytest.mark.integration
def test_cli_generate_command():
    """Integration test for CLI generate command."""
    pass

@pytest.mark.performance
def test_large_file_processing():
    """Performance test for large file processing."""
    pass
```

### Writing Tests

#### Unit Test Example
```python
import pytest
from unittest.mock import Mock, patch
from testgen_copilot.generator import TestGenerator

class TestTestGenerator:
    def setup_method(self):
        self.generator = TestGenerator()
    
    def test_generate_simple_function(self):
        """Test generation for a simple function."""
        source_code = """
        def add(a, b):
            return a + b
        """
        
        result = self.generator.generate(source_code)
        
        assert len(result.test_cases) > 0
        assert "test_add" in result.test_cases[0].name
        assert "assert" in result.test_cases[0].code
```

#### Integration Test Example
```python
import tempfile
from pathlib import Path
from click.testing import CliRunner
from testgen_copilot.cli import main

def test_generate_command_creates_test_file():
    """Test that generate command creates test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_file = Path(temp_dir) / "example.py"
        source_file.write_text("def hello(): return 'world'")
        
        output_dir = Path(temp_dir) / "tests"
        output_dir.mkdir()
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'generate',
            '--file', str(source_file),
            '--output', str(output_dir)
        ])
        
        assert result.exit_code == 0
        test_files = list(output_dir.glob("test_*.py"))
        assert len(test_files) > 0
```

### Test Configuration

#### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
```

#### Fixtures
```python
# conftest.py
import pytest
from testgen_copilot.config import Config

@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return Config(
        language="python",
        test_framework="pytest",
        coverage_target=0.8
    )

@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    
    return {
        "root": tmp_path,
        "src": src_dir,
        "tests": test_dir
    }
```

## Documentation

### Documentation Structure

```
docs/
├── index.md             # Main documentation page
├── getting-started.md   # Installation and quick start
├── user-guide/         # User documentation
├── api/                # API reference
├── development/        # Development documentation
├── examples/           # Code examples
└── _static/           # Static assets
```

### Writing Documentation

#### Docstrings
```python
def analyze_code(source_code: str, language: str = "python") -> AnalysisResult:
    """Analyze source code for potential issues.
    
    This function performs comprehensive static analysis including
    security vulnerability detection, code quality assessment,
    and complexity analysis.
    
    Args:
        source_code: The source code to analyze
        language: Programming language of the source code
        
    Returns:
        AnalysisResult containing findings and recommendations
        
    Raises:
        UnsupportedLanguageError: If the language is not supported
        ParseError: If the source code cannot be parsed
        
    Example:
        >>> result = analyze_code("def hello(): pass", "python")
        >>> print(result.issues)
        []
    """
```

#### Markdown Documentation
- Use clear headings and structure
- Include code examples with syntax highlighting
- Add links to related sections
- Include screenshots for UI features

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve documentation locally
make docs-serve

# Check for broken links
make docs-check
```

## Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Workflow

1. **Update version in pyproject.toml**
2. **Update CHANGELOG.md**
3. **Create release PR**
4. **Merge after approval**
5. **Tag release**
6. **Automated CI/CD handles publishing**

### Creating a Release

```bash
# Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or use the release workflow
gh workflow run release.yml -f version=v1.0.0
```

### Changelog Format

```markdown
## [1.0.0] - 2024-01-15

### Added
- New feature X
- Support for language Y

### Changed
- Improved performance of Z
- Updated API for better usability

### Fixed
- Bug in feature A
- Security vulnerability in component B

### Removed
- Deprecated feature C
```

## Troubleshooting

### Common Issues

#### Development Environment

**Issue**: Import errors when running tests
```bash
# Solution: Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Issue**: Pre-commit hooks failing
```bash
# Solution: Update and reinstall hooks
pre-commit autoupdate
pre-commit install --overwrite
```

#### Testing

**Issue**: Tests fail with module not found
```bash
# Solution: Install package in development mode
pip install -e .
```

**Issue**: Coverage reports incomplete
```bash
# Solution: Run with proper coverage configuration
pytest --cov=src/testgen_copilot --cov-config=pyproject.toml
```

#### Docker

**Issue**: Permission denied in Docker container
```bash
# Solution: Check user permissions and volume mounts
docker run --user $(id -u):$(id -g) ...
```

### Debugging

#### Logging
```python
import logging
from testgen_copilot.logging_config import setup_logging

# Enable debug logging
setup_logging(level=logging.DEBUG)

# Or use CLI
testgen --log-level debug generate --file example.py
```

#### Profiling
```bash
# Profile performance
python -m cProfile -o profile.stats -m testgen_copilot.cli generate --file example.py

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### Getting Help

- **Documentation**: Check the user guide and API reference
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our community forums

### Contributing Guidelines

1. **Follow the code style guidelines**
2. **Write comprehensive tests**
3. **Update documentation**
4. **Use conventional commit messages**
5. **Keep pull requests focused and small**

For more detailed contributing guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).