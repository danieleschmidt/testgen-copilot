# Development Guide

## Getting Started

This guide will help you set up a development environment and contribute to TestGen Copilot Assistant.

### Prerequisites

- Python 3.8+ (3.11 recommended)
- Git
- Docker (optional, for containerized development)
- VS Code (recommended)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/testgen/copilot-assistant.git
   cd copilot-assistant
   ```

2. **Set up development environment**:
   ```bash
   make setup-dev
   ```

3. **Verify installation**:
   ```bash
   testgen --version
   make test
   ```

### Development Environment Options

#### Option 1: Local Development

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev,ai]"
   pre-commit install
   ```

#### Option 2: Dev Container (Recommended)

1. **Open in VS Code**:
   ```bash
   code .
   ```

2. **Use Dev Container**:
   - Press `F1` → "Dev Containers: Reopen in Container"
   - Everything is pre-configured!

#### Option 3: Docker Development

1. **Build development image**:
   ```bash
   docker-compose build testgen
   ```

2. **Start development environment**:
   ```bash
   docker-compose up testgen
   ```

## Project Structure

```
testgen-copilot-assistant/
├── .devcontainer/          # Development container configuration
├── .github/                # GitHub workflows and templates
├── .vscode/                # VS Code settings
├── docs/                   # Documentation
│   ├── guides/            # User guides
│   ├── runbooks/          # Operational runbooks
│   └── api/               # API documentation
├── monitoring/            # Monitoring configuration
├── scripts/               # Utility scripts
├── src/testgen_copilot/   # Main application code
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── core.py            # Core functionality
│   ├── generator.py       # Test generation
│   ├── security.py        # Security scanning
│   ├── coverage.py        # Coverage analysis
│   └── ...
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── e2e/              # End-to-end tests
│   ├── performance/      # Performance tests
│   └── fixtures/         # Test data
├── pyproject.toml        # Project configuration
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Development services
├── Makefile             # Development commands
└── README.md            # Project overview
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Run tests**:
   ```bash
   make test
   ```

4. **Run quality checks**:
   ```bash
   make quality
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Coding Standards

#### Python Style

- **Formatter**: Black (88 character line length)
- **Linter**: Ruff with strict settings
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public APIs
- **Import sorting**: isort with Black compatibility

#### Code Quality

- **Test Coverage**: Minimum 85% line coverage
- **Security**: All code scanned with Bandit
- **Type Checking**: MyPy in strict mode
- **Documentation**: All public APIs documented

#### Git Conventions

- **Commit Messages**: Follow conventional commits
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `test:` for tests
  - `refactor:` for refactoring
  - `chore:` for maintenance

### Testing Strategy

#### Unit Tests

Located in `tests/unit/`, these test individual functions and classes:

```bash
# Run unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=src/testgen_copilot
```

#### Integration Tests

Located in `tests/integration/`, these test component interactions:

```bash
# Run integration tests
pytest tests/integration/

# Run with services
docker-compose up -d redis
pytest tests/integration/
```

#### End-to-End Tests

Located in `tests/e2e/`, these test complete workflows:

```bash
# Run E2E tests
pytest tests/e2e/

# Run against staging
TESTGEN_ENV=staging pytest tests/e2e/
```

#### Performance Tests

Located in `tests/performance/`, these test performance characteristics:

```bash
# Run performance tests
pytest tests/performance/ --benchmark-only

# Save benchmark results
pytest tests/performance/ --benchmark-save=baseline
```

### Common Development Tasks

#### Adding a New Command

1. **Add command to CLI**:
   ```python
   # src/testgen_copilot/cli.py
   @click.command()
   def new_command():
       """Your new command description."""
       pass
   ```

2. **Implement functionality**:
   ```python
   # src/testgen_copilot/new_feature.py
   def new_feature_logic():
       """Implement your feature here."""
       pass
   ```

3. **Add tests**:
   ```python
   # tests/unit/test_new_feature.py
   def test_new_feature():
       """Test your new feature."""
       pass
   ```

#### Adding a New Security Rule

1. **Define the rule**:
   ```python
   # src/testgen_copilot/security_rules.py
   class NewSecurityRule(SecurityRule):
       def check(self, code):
           # Your security check logic
           pass
   ```

2. **Register the rule**:
   ```python
   # src/testgen_copilot/security.py
   SECURITY_RULES.append(NewSecurityRule())
   ```

3. **Add tests**:
   ```python
   # tests/unit/test_security.py
   def test_new_security_rule():
       # Test your security rule
       pass
   ```

### Debugging

#### Local Debugging

1. **Enable debug logging**:
   ```bash
   export TESTGEN_LOG_LEVEL=DEBUG
   testgen your-command
   ```

2. **Use pdb for breakpoints**:
   ```python
   import pdb; pdb.set_trace()
   ```

3. **VS Code debugging**: Use the included launch configurations

#### Container Debugging

1. **Attach to running container**:
   ```bash
   docker-compose exec testgen bash
   ```

2. **Debug with remote debugging**:
   ```python
   import debugpy
   debugpy.listen(("0.0.0.0", 5678))
   debugpy.wait_for_client()
   ```

### Performance Profiling

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile a command
mprof run testgen generate --file large_file.py
mprof plot
```

#### CPU Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.prof -m testgen_copilot.cli generate --file test.py

# Analyze results
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"
```

### Documentation

#### API Documentation

- **Docstrings**: Use Google style
- **Type hints**: Include for all parameters and returns
- **Examples**: Provide usage examples

```python
def generate_tests(file_path: str, output_dir: str = "tests") -> List[str]:
    """Generate test files for the given source file.
    
    Args:
        file_path: Path to the source file to analyze
        output_dir: Directory to write generated tests
        
    Returns:
        List of generated test file paths
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        
    Example:
        >>> generate_tests("src/calculator.py", "tests/")
        ["tests/test_calculator.py"]
    """
```

#### User Documentation

- **Guides**: Step-by-step tutorials
- **Reference**: Complete command documentation
- **Examples**: Real-world usage scenarios

### Release Process

#### Version Management

Versions follow semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

#### Creating a Release

1. **Update version**:
   ```bash
   # Version is managed automatically by semantic-release
   git commit -m "feat: your new feature"
   git push origin main
   ```

2. **Create release** (automated via CI/CD):
   - Tags are created automatically
   - Release notes generated from commits
   - Packages published to PyPI

### Troubleshooting

#### Common Issues

1. **Import errors**:
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   ```

2. **Test failures**:
   ```bash
   # Clean test cache
   pytest --cache-clear
   
   # Recreate virtual environment
   deactivate
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Docker issues**:
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

#### Getting Help

- **Documentation**: Check this guide and other docs
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our Discord/Slack (links in README)

### Contributing Guidelines

#### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass (`make test`)
- [ ] Quality checks pass (`make quality`)
- [ ] Documentation is updated
- [ ] CHANGELOG is updated (if applicable)

#### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Document** your changes
6. **Submit** a pull request

#### Code Review

- All PRs require review
- Address reviewer feedback
- Maintain clean commit history
- Squash commits if requested

### Advanced Topics

#### Custom Integrations

See `docs/guides/custom-integrations.md` for creating custom:
- Test framework adapters
- Security rule engines
- Coverage analyzers
- Output formatters

#### Performance Optimization

- Use profiling tools to identify bottlenecks
- Implement caching for expensive operations
- Consider parallel processing for large projects
- Monitor memory usage for large codebases

#### Security Considerations

- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs
- Follow security best practices guide

## Next Steps

- Read the [User Guide](USAGE.md)
- Explore [API Documentation](api/)
- Check out [Examples](guides/examples.md)
- Join the [Community](https://discord.gg/testgen)