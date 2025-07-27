# TestGen Copilot Assistant API Documentation

## Overview

The TestGen Copilot Assistant provides both a Command Line Interface (CLI) and a Python API for automated test generation and security analysis. This document covers the Python API usage.

## Installation

```bash
pip install testgen-copilot-assistant
```

## Quick Start

```python
from testgen_copilot import TestGenerator, SecurityScanner, CoverageAnalyzer

# Initialize the test generator
generator = TestGenerator(language='python')

# Generate tests for a Python file
tests = generator.generate_tests('src/calculator.py')
print(tests)
```

## Core Classes

### TestGenerator

The main class for generating comprehensive unit tests.

```python
class TestGenerator:
    def __init__(
        self,
        language: str = 'python',
        test_framework: str = 'pytest',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TestGenerator.
        
        Args:
            language: Programming language ('python', 'javascript', 'typescript', etc.)
            test_framework: Testing framework ('pytest', 'unittest', 'jest', etc.)
            config: Optional configuration dictionary
        """
```

#### Methods

##### generate_tests()

Generate comprehensive tests for a source file.

```python
def generate_tests(
    self,
    file_path: str,
    output_path: Optional[str] = None,
    include_edge_cases: bool = True,
    include_error_tests: bool = True,
    include_mocking: bool = True
) -> str:
    """
    Generate comprehensive tests for a source file.
    
    Args:
        file_path: Path to the source file to analyze
        output_path: Optional path to write generated tests
        include_edge_cases: Whether to include edge case tests
        include_error_tests: Whether to include error path tests
        include_mocking: Whether to include mocking scenarios
    
    Returns:
        Generated test code as string
    
    Raises:
        FileNotFoundError: If source file doesn't exist
        ValueError: If language/framework not supported
    """
```

##### generate_batch()

Generate tests for multiple files at once.

```python
def generate_batch(
    self,
    source_dir: str,
    output_dir: str,
    patterns: List[str] = None,
    parallel: bool = True
) -> Dict[str, str]:
    """
    Generate tests for multiple files.
    
    Args:
        source_dir: Directory containing source files
        output_dir: Directory to write test files
        patterns: File patterns to include (e.g., ['*.py', '*.js'])
        parallel: Whether to process files in parallel
    
    Returns:
        Dictionary mapping source files to generated test content
    """
```

### SecurityScanner

Analyze code for potential security vulnerabilities.

```python
class SecurityScanner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SecurityScanner.
        
        Args:
            config: Optional configuration for security rules
        """
```

#### Methods

##### scan_file()

Scan a single file for security issues.

```python
def scan_file(self, file_path: str) -> SecurityReport:
    """
    Scan a file for security vulnerabilities.
    
    Args:
        file_path: Path to file to scan
    
    Returns:
        SecurityReport with findings and recommendations
    """
```

##### scan_project()

Scan entire project for security issues.

```python
def scan_project(
    self,
    project_path: str,
    exclude_patterns: List[str] = None
) -> ProjectSecurityReport:
    """
    Scan entire project for security issues.
    
    Args:
        project_path: Path to project root
        exclude_patterns: Patterns to exclude from scanning
    
    Returns:
        ProjectSecurityReport with aggregated findings
    """
```

### CoverageAnalyzer

Analyze test coverage and provide insights.

```python
class CoverageAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CoverageAnalyzer."""
```

#### Methods

##### analyze()

Analyze test coverage for a project.

```python
def analyze(
    self,
    source_dir: str,
    tests_dir: str = 'tests'
) -> CoverageReport:
    """
    Analyze test coverage.
    
    Args:
        source_dir: Directory containing source code
        tests_dir: Directory containing tests
    
    Returns:
        CoverageReport with coverage metrics and gaps
    """
```

### QualityScorer

Evaluate test quality and effectiveness.

```python
class QualityScorer:
    def __init__(self):
        """Initialize QualityScorer."""
```

#### Methods

##### score()

Score test quality for a test suite.

```python
def score(self, tests_dir: str) -> QualityReport:
    """
    Score test quality.
    
    Args:
        tests_dir: Directory containing test files
    
    Returns:
        QualityReport with quality metrics and recommendations
    """
```

## Data Models

### SecurityReport

```python
@dataclass
class SecurityReport:
    file_path: str
    issues: List[SecurityIssue]
    severity_counts: Dict[str, int]
    overall_score: float
    recommendations: List[str]
```

### SecurityIssue

```python
@dataclass
class SecurityIssue:
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    line_number: int
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
```

### CoverageReport

```python
@dataclass
class CoverageReport:
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    missing_lines: List[int]
    uncovered_functions: List[str]
    total_lines: int
    covered_lines: int
```

### QualityReport

```python
@dataclass
class QualityReport:
    overall_score: float
    test_count: int
    assertion_count: int
    mocking_usage: float
    edge_case_coverage: float
    error_path_coverage: float
    recommendations: List[str]
```

## Configuration

### Configuration File

Create a `.testgen.config.json` file in your project root:

```json
{
  "language": "python",
  "test_framework": "pytest",
  "coverage_target": 85,
  "security_rules": {
    "sql_injection": true,
    "xss_vulnerabilities": true,
    "authentication_bypass": true,
    "data_exposure": true,
    "unsafe_deserialization": true,
    "path_traversal": true,
    "command_injection": true,
    "crypto_misuse": true
  },
  "test_patterns": {
    "edge_cases": true,
    "error_handling": true,
    "mocking": true,
    "integration_scenarios": false,
    "performance_tests": false,
    "benchmark_tests": false
  },
  "output": {
    "format": "standard",
    "include_docstrings": true,
    "add_comments": true,
    "type_hints": true
  },
  "ai_providers": {
    "primary": "openai",
    "fallback": "anthropic",
    "timeout": 30
  }
}
```

### Environment Variables

```bash
# AI API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# TestGen Configuration
export TESTGEN_ENV="development"
export TESTGEN_LOG_LEVEL="INFO"
export TESTGEN_CACHE_DIR="~/.testgen/cache"
export TESTGEN_CONFIG_FILE=".testgen.config.json"
```

## Examples

### Basic Test Generation

```python
from testgen_copilot import TestGenerator

# Initialize generator
generator = TestGenerator(language='python', test_framework='pytest')

# Generate tests for a calculator module
test_code = generator.generate_tests(
    file_path='src/calculator.py',
    include_edge_cases=True,
    include_error_tests=True
)

# Write tests to file
with open('tests/test_calculator.py', 'w') as f:
    f.write(test_code)

print("Tests generated successfully!")
```

### Security Scanning

```python
from testgen_copilot import SecurityScanner

# Initialize scanner
scanner = SecurityScanner()

# Scan a single file
report = scanner.scan_file('src/user_auth.py')

print(f"Security Score: {report.overall_score}/100")
print(f"Issues Found: {len(report.issues)}")

for issue in report.issues:
    print(f"⚠️  {issue.severity.upper()}: {issue.description}")
    print(f"    Line {issue.line_number}: {issue.recommendation}")

# Scan entire project
project_report = scanner.scan_project('.')
print(f"Project Security Score: {project_report.overall_score}/100")
```

### Coverage Analysis

```python
from testgen_copilot import CoverageAnalyzer

analyzer = CoverageAnalyzer()

# Analyze coverage
coverage = analyzer.analyze(source_dir='src', tests_dir='tests')

print(f"Line Coverage: {coverage.line_coverage:.1f}%")
print(f"Branch Coverage: {coverage.branch_coverage:.1f}%")
print(f"Function Coverage: {coverage.function_coverage:.1f}%")

if coverage.missing_lines:
    print(f"Missing lines: {coverage.missing_lines}")

if coverage.uncovered_functions:
    print(f"Uncovered functions: {coverage.uncovered_functions}")
```

### Quality Assessment

```python
from testgen_copilot import QualityScorer

scorer = QualityScorer()

# Score test quality
quality = scorer.score('tests')

print(f"Test Quality Score: {quality.overall_score:.1f}/100")
print(f"Total Tests: {quality.test_count}")
print(f"Edge Case Coverage: {quality.edge_case_coverage:.1f}%")
print(f"Error Path Coverage: {quality.error_path_coverage:.1f}%")

for recommendation in quality.recommendations:
    print(f"💡 {recommendation}")
```

### Batch Processing

```python
from testgen_copilot import TestGenerator
import os

generator = TestGenerator()

# Generate tests for all Python files in src/
results = generator.generate_batch(
    source_dir='src',
    output_dir='tests',
    patterns=['*.py'],
    parallel=True
)

print(f"Generated tests for {len(results)} files:")
for source_file, test_content in results.items():
    print(f"  {source_file} -> tests/test_{os.path.basename(source_file)}")
```

### Custom Configuration

```python
from testgen_copilot import TestGenerator

# Custom configuration
config = {
    "language": "python",
    "test_framework": "pytest",
    "coverage_target": 95,
    "test_patterns": {
        "edge_cases": True,
        "error_handling": True,
        "mocking": True,
        "performance_tests": True
    },
    "ai_providers": {
        "primary": "anthropic",
        "timeout": 60
    }
}

generator = TestGenerator(config=config)

# Generate tests with custom config
tests = generator.generate_tests('src/complex_module.py')
```

## Error Handling

The API uses standard Python exceptions:

```python
from testgen_copilot import TestGenerator, TestGenError

try:
    generator = TestGenerator()
    tests = generator.generate_tests('nonexistent.py')
except FileNotFoundError:
    print("Source file not found")
except TestGenError as e:
    print(f"TestGen error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Async Support

For I/O intensive operations, use the async API:

```python
import asyncio
from testgen_copilot import AsyncTestGenerator

async def generate_tests_async():
    generator = AsyncTestGenerator()
    
    # Generate tests asynchronously
    tests = await generator.generate_tests('src/calculator.py')
    return tests

# Run async
tests = asyncio.run(generate_tests_async())
```

## Integration with Testing Frameworks

### pytest Integration

```python
# conftest.py
import pytest
from testgen_copilot import TestGenerator

@pytest.fixture
def test_generator():
    return TestGenerator(language='python', test_framework='pytest')

@pytest.fixture(autouse=True)
def auto_generate_tests(request, test_generator):
    """Automatically generate tests for modules that don't have them."""
    if not hasattr(request.module, '__file__'):
        return
    
    source_file = request.module.__file__.replace('test_', '').replace('tests/', 'src/')
    if not os.path.exists(source_file):
        return
    
    # Generate tests if they don't exist
    test_file = request.module.__file__
    if not os.path.exists(test_file):
        tests = test_generator.generate_tests(source_file)
        with open(test_file, 'w') as f:
            f.write(tests)
```

### unittest Integration

```python
import unittest
from testgen_copilot import TestGenerator

class TestAutoGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = TestGenerator(test_framework='unittest')
    
    def test_calculator_module(self):
        """Auto-generated tests for calculator module."""
        tests = self.generator.generate_tests('src/calculator.py')
        
        # Execute generated tests
        exec(tests, globals())
```

## API Reference Summary

| Class | Primary Use | Key Methods |
|-------|-------------|-------------|
| `TestGenerator` | Test generation | `generate_tests()`, `generate_batch()` |
| `SecurityScanner` | Security analysis | `scan_file()`, `scan_project()` |
| `CoverageAnalyzer` | Coverage analysis | `analyze()` |
| `QualityScorer` | Test quality assessment | `score()` |

## Support

For API support and examples:

- [GitHub Issues](https://github.com/testgen/copilot-assistant/issues)
- [Documentation](https://testgen.readthedocs.io)
- [Examples Repository](https://github.com/testgen/copilot-examples)