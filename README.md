# TestGen-Copilot-Assistant

CLI tool and VS Code extension that uses LLMs to automatically generate comprehensive unit tests and highlight potential security vulnerabilities in your codebase.

## Features

- **Intelligent Test Generation**: Creates comprehensive unit tests with edge cases and mocking
- **Security Vulnerability Detection**: Identifies potential security flaws and suggests fixes
- **Multi-Language Support**: Python, JavaScript/TypeScript, Java, C#, Go, and Rust
- **IDE Integration**: Native VS Code extension with real-time suggestions
- **Coverage Analysis**: Ensures generated tests achieve high code coverage
- **Test Quality Scoring**: Evaluates test effectiveness and completeness

## Installation

### CLI Tool
```bash
pip install testgen-copilot-assistant
# or
npm install -g testgen-copilot-assistant
```

### VS Code Extension
Search for "TestGen Copilot Assistant" in the VS Code marketplace or install via:
```bash
code --install-extension testgen.copilot-assistant
```

## Quick Start

### Command Line Usage
```bash
# Generate tests for a single file
testgen --file src/calculator.py --output tests/

# Generate tests for every file in a project
testgen --project . --output tests --batch  # requires --project and --output only

# Use a configuration file
testgen --config myconfig.json --file src/calculator.py --output tests
# A file named `.testgen.config.json` in the current or project directory
# is loaded automatically when present

# Analyze entire project and enforce 90% coverage
testgen --project . --security-scan --coverage-target 90

# Check coverage only (no test generation)
# default tests directory is 'tests'
testgen --project . --coverage-target 80

# Use a custom tests directory
testgen --project . --coverage-target 80 --tests-dir mytests

# Show missing functions when checking coverage
testgen --project . --coverage-target 80 --show-missing

# Enforce test quality score
testgen --project . --quality-target 90

# Skip edge case tests
testgen --file src/calculator.py --output tests --no-edge-cases

# Skip error path tests
testgen --file src/calculator.py --output tests --no-error-tests

# Skip benchmark tests
testgen --file src/calculator.py --output tests --no-benchmark-tests

# Skip integration tests
testgen --file src/calculator.py --output tests --no-integration-tests

# Watch mode for continuous testing
# pass `--auto-generate` to write tests automatically
# adjust polling interval with --poll (seconds)
testgen --watch src/ --output tests --auto-generate --poll 2.0
```

### VS Code Integration
1. Open any source file
2. Right-click → "Generate Tests with TestGen"
3. Review generated tests in the side panel
4. Accept, modify, or regenerate as needed

## Configuration

Create `.testgen.config.json` in your project root:

```json
{
  "language": "python",
  "test_framework": "pytest",
  "coverage_target": 85,
  "security_rules": {
    "sql_injection": true,
    "xss_vulnerabilities": true,
    "authentication_bypass": true,
    "data_exposure": true
  },
  "test_patterns": {
    "edge_cases": true,
    "error_handling": true,
    "mocking": true,
    "integration_scenarios": false  # disable integration tests
  },
  "output": {
    "format": "standard",
    "include_docstrings": true,
    "add_comments": true
  }
}
```

## Example Output

### Input Code
```python
def calculate_discount(price, discount_percent):
    return price * (1 - discount_percent / 100)
```

### Generated Tests
```python
import pytest
from calculator import calculate_discount

class TestCalculateDiscount:
    """Comprehensive tests for calculate_discount function."""
    
    def test_basic_discount_calculation(self):
        """Test standard discount calculation."""
        result = calculate_discount(100, 10)
        assert result == 90.0
    
    def test_zero_discount(self):
        """Test with zero discount."""
        result = calculate_discount(100, 0)
        assert result == 100.0
    
    def test_full_discount(self):
        """Test with 100% discount."""
        result = calculate_discount(100, 100)
        assert result == 0.0
    
    def test_negative_price_edge_case(self):
        """Test behavior with negative price."""
        result = calculate_discount(-50, 10)
        assert result == -45.0
    
    def test_discount_over_100_percent(self):
        """Test edge case with discount over 100%."""
        result = calculate_discount(100, 150)
        assert result == -50.0  # May indicate business logic issue
    
    @pytest.mark.parametrize("price,discount,expected", [
        (200, 25, 150),
        (50, 50, 25),
        (1000, 5, 950)
    ])
    def test_various_discount_scenarios(self, price, discount, expected):
        """Test multiple discount scenarios."""
        assert calculate_discount(price, discount) == expected
```

### Security Analysis
```markdown
## Security Analysis Report

### ⚠️ Potential Issues Found:
1. **Input Validation Missing**: Function doesn't validate discount_percent range
2. **Business Logic Flaw**: Allows discounts > 100%, could lead to negative prices
3. **Type Safety**: No type checking on inputs could cause runtime errors

### 🛡️ Recommendations:
- Add input validation: `if not 0 <= discount_percent <= 100:`
- Consider raising ValueError for invalid inputs
- Add type hints: `def calculate_discount(price: float, discount_percent: float) -> float:`
```

## Features

### Test Generation Capabilities
- **Unit Tests**: Comprehensive test suites with fixtures and mocks
- **Edge Case Detection**: Automatically identifies boundary conditions
- **Error Path Testing**: Tests exception handling and error states
- **Performance Tests**: Basic benchmark tests for critical functions
- **Integration Tests**: Optional cross-module testing scenarios

### Security Analysis
- **OWASP Top 10**: Scans for common web vulnerabilities
- **Input Validation**: Identifies missing or weak input validation
- **Authentication Issues**: Detects authentication bypass possibilities
- **Data Exposure**: Finds potential information leakage
- **Injection Attacks**: SQL, NoSQL, and command injection detection

### IDE Features
- **Real-time Generation**: Tests generated as you type
- **Inline Suggestions**: Security warnings directly in code
- **Test Coverage Visualization**: Shows coverage gaps in real-time
- **One-click Fixes**: Apply suggested security improvements
- **Batch Processing**: Generate tests for entire projects

## Supported Frameworks

### Testing Frameworks
- **Python**: pytest, unittest, nose2
- **JavaScript**: Jest, Mocha, Jasmine, Vitest
- **TypeScript**: Jest, Vitest, Deno
- **Java**: JUnit 5, TestNG, Mockito
- **C#**: NUnit, MSTest, xUnit
- **Go**: testing package, Testify
- **Rust**: built-in test framework

### Language-Specific Features
Each language integration includes:
- Framework-specific test patterns
- Appropriate mocking libraries
- Language idiom compliance
- Standard assertion libraries

## Advanced Usage

### Custom Test Templates
```bash
# Create custom test template
testgen --create-template python-api-tests

# Use custom template
testgen --template python-api-tests --file api.py
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Generate and run tests
  run: |
    testgen --project . --ci-mode
    pytest --cov=src tests/
```

### API Integration
```python
from testgen import TestGenerator

generator = TestGenerator(language='python')
tests = generator.generate_tests('src/calculator.py')
security_report = generator.analyze_security('src/')
```

### Coverage Analysis
```python
from testgen_copilot import CoverageAnalyzer

analyzer = CoverageAnalyzer()
percent = analyzer.analyze('src/calculator.py', 'tests')  # or any tests directory
print(f"Calculator module covered: {percent:.1f}%")
```

### Test Quality Scoring
```python
from testgen_copilot import TestQualityScorer

scorer = TestQualityScorer()
quality = scorer.score('tests')
print(f"Test suite quality: {quality:.1f}%")
```

Use `--quality-target` on the CLI to enforce a minimum score:
```bash
testgen --project . --quality-target 90
```

## Contributing

We welcome contributions in the following areas:
- Additional language support
- New security rule implementations
- Test framework integrations
- IDE plugin development
- Performance improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Roadmap

- [ ] AI-powered test maintenance and updates
- [ ] Visual test coverage reporting
- [ ] Integration with popular CI/CD platforms
- [ ] Advanced security vulnerability database
- [ ] Machine learning-based test quality assessment
- [ ] Support for additional IDEs (IntelliJ, Vim, Emacs)

## License

MIT License - see [LICENSE](LICENSE) file for details.
