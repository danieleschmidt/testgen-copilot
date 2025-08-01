# Testing Documentation

This document outlines the comprehensive testing strategy and infrastructure for TestGen-Copilot.

## Testing Architecture

### Test Categories

#### Unit Tests (`tests/test_*.py`)
- Test individual functions and classes in isolation
- Fast execution (< 1 second per test)
- Mock external dependencies
- Located in root `tests/` directory

#### Integration Tests (`tests/integration/`)
- Test interaction between components
- Test external API integrations
- Database and file system interactions
- Slower execution (1-10 seconds per test)

#### End-to-End Tests (`tests/e2e/`)
- Test complete user workflows
- CLI command testing
- VS Code extension integration
- Slowest execution (10+ seconds per test)

#### Performance Tests (`tests/performance/`)
- Benchmark critical code paths
- Memory usage validation
- Timeout testing
- Load testing scenarios

#### Security Tests (`tests/security/`)
- Vulnerability detection validation
- Input sanitization testing
- Authentication/authorization tests
- Security rule accuracy testing

#### Mutation Tests (`tests/mutation/`)
- Test quality validation using mutation testing
- Ensures tests catch actual bugs
- Validates test suite effectiveness

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data and sample files
│   ├── sample_code.py       # Sample code for testing
│   └── test_data.json       # JSON test data
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── e2e/                     # End-to-end tests
├── performance/             # Performance and load tests
├── mutation/                # Mutation testing
└── load/                    # Load testing with Locust
    └── locustfile.py        # Load testing scenarios
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/testgen_copilot

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m security
pytest -m performance

# Run tests in parallel
pytest -n auto

# Run with detailed output
pytest -v

# Run specific test file
pytest tests/test_cli.py

# Run specific test function
pytest tests/test_cli.py::test_generate_command
```

### Advanced Test Options

```bash
# Run slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run only smoke tests
pytest -m smoke

# Generate HTML coverage report
pytest --cov=src/testgen_copilot --cov-report=html

# Run with profiling
pytest --profile

# Run with mutation testing
mutmut run

# Load testing
locust -f tests/load/locustfile.py
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

The pytest configuration includes:
- Coverage reporting with 85% minimum threshold
- Branch coverage enabled
- HTML and XML report generation
- Parallel test execution
- Strict marker validation
- Comprehensive warning filters

### Coverage Configuration

Coverage is configured to:
- Cover `src/testgen_copilot` package
- Generate HTML, XML, and terminal reports
- Require minimum 85% coverage
- Include branch coverage
- Exclude test files from coverage

## Test Fixtures

### Global Fixtures (`conftest.py`)

Common fixtures available to all tests:

```python
@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
def calculate_sum(a, b):
    return a + b
'''

@pytest.fixture
def temp_project_dir(tmp_path):
    """Temporary project directory with sample files."""
    # Creates realistic project structure for testing

@pytest.fixture
def mock_llm_response():
    """Mock LLM API responses for testing."""
    # Provides consistent LLM responses for tests
```

### Test Data

Test data is organized in `tests/fixtures/`:
- `sample_code.py`: Various code samples for different languages
- `test_data.json`: JSON configuration and test data
- Language-specific sample files for multi-language testing

## Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit
def test_function_behavior():
    """Unit test example."""
    pass

@pytest.mark.integration
def test_api_integration():
    """Integration test example."""
    pass

@pytest.mark.slow
def test_large_file_processing():
    """Slow test example."""
    pass

@pytest.mark.security
def test_vulnerability_detection():
    """Security test example."""
    pass
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Scheduled nightly runs

CI includes:
- Multi-platform testing (Linux, macOS, Windows)
- Multi-Python version testing (3.8, 3.9, 3.10, 3.11)
- Coverage reporting
- Security scanning
- Performance regression detection

### Test Reports

CI generates:
- JUnit XML reports for test results
- HTML coverage reports
- Performance benchmarks
- Security scan results

## Writing Tests

### Test Naming Convention

```python
def test_should_generate_unit_tests_when_given_valid_python_code():
    """Test function names should be descriptive and follow should/when pattern."""
    pass

def test_generate_tests_invalid_input_raises_validation_error():
    """Alternative naming pattern for error cases."""
    pass
```

### Test Structure (AAA Pattern)

```python
def test_calculate_discount():
    # Arrange
    calculator = PriceCalculator()
    price = 100.0
    discount_percent = 10.0
    
    # Act
    result = calculator.calculate_discount(price, discount_percent)
    
    # Assert
    assert result == 90.0
```

### Mocking External Dependencies

```python
@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client for testing."""
    mock_client = mocker.patch('openai.Client')
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="generated test code"))]
    )
    return mock_client

def test_generate_tests_with_openai(mock_openai_client):
    generator = TestGenerator()
    result = generator.generate_tests("def add(a, b): return a + b")
    assert "test_add" in result
```

## Performance Testing

### Benchmark Tests

```python
def test_generation_performance(benchmark):
    """Benchmark test generation performance."""
    generator = TestGenerator()
    result = benchmark(generator.generate_tests, sample_code)
    assert len(result) > 0
```

### Load Testing

Use Locust for load testing:

```bash
# Start load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Run headless load test
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 1m --headless
```

## Mutation Testing

Mutation testing validates test quality:

```bash
# Run mutation testing
mutmut run

# Show mutation results
mutmut results

# Show specific mutations
mutmut show
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/`
2. **Coverage Issues**: Check file paths in coverage configuration
3. **Slow Tests**: Use `-m "not slow"` to skip slow tests during development
4. **Flaky Tests**: Mark with `@pytest.mark.flaky` and investigate timing issues

### Debug Options

```bash
# Run with debugging
pytest --pdb

# Verbose output
pytest -vv

# Show local variables in tracebacks
pytest -l

# Stop on first failure
pytest -x
```

## Best Practices

### Test Design
- Write tests before implementing features (TDD)
- Keep tests simple and focused
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test both happy path and error cases

### Test Maintenance
- Regularly review and update tests
- Remove obsolete tests
- Refactor common test code into fixtures
- Monitor test execution time
- Keep test data up to date

### Coverage Guidelines
- Aim for 90%+ line coverage
- Focus on branch coverage for complex logic
- Don't chase 100% coverage at expense of test quality
- Exclude trivial code from coverage requirements
- Regular coverage report review

---

For questions about testing, see the [CONTRIBUTING.md](../CONTRIBUTING.md) guide or reach out to the development team.