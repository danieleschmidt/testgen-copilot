"""Pytest configuration and custom plugins."""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "smoke: Basic smoke tests for quick validation"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests for bug fixes"
    )
    config.addinivalue_line(
        "markers", "acceptance: User acceptance tests"
    )
    
    # Set test environment variables
    os.environ["TESTGEN_ENV"] = "testing"
    os.environ["TESTGEN_DEV_MODE"] = "true"
    os.environ["TESTGEN_LOG_LEVEL"] = "DEBUG"
    os.environ["TESTGEN_CACHE_ENABLED"] = "false"
    
    # Disable external API calls during testing
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or hasattr(item.function, "_pytest_mark_slow"):
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Auto-mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)


def pytest_runtest_setup(item):
    """Setup hook for individual test runs."""
    # Skip tests requiring network if offline
    if item.get_closest_marker("network") and os.environ.get("PYTEST_OFFLINE"):
        pytest.skip("Network tests skipped in offline mode")
    
    # Skip tests requiring API keys if not available
    if item.get_closest_marker("api"):
        required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        if not any(os.environ.get(key, "").startswith("sk-") for key in required_keys):
            if not os.environ.get("PYTEST_ALLOW_MOCK_API"):
                pytest.skip("API tests require valid API keys or PYTEST_ALLOW_MOCK_API=1")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")  
def temp_workspace() -> Generator[Path, None, None]:
    """Provide a temporary workspace for tests."""
    with tempfile.TemporaryDirectory(prefix="testgen_test_") as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = """
import pytest

def test_sample_function():
    '''Test the sample function.'''
    result = sample_function(1, 2)
    assert result == 3
    
def test_sample_function_edge_cases():
    '''Test edge cases for sample function.'''
    assert sample_function(0, 0) == 0
    assert sample_function(-1, 1) == 0
"""
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for testing."""
    return """
def sample_function(a, b):
    '''Add two numbers together.'''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    return a + b

class Calculator:
    '''Simple calculator class.'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        '''Add two numbers.'''
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        '''Get calculation history.'''
        return self.history.copy()
"""


@pytest.fixture
def sample_javascript_code():
    """Provide sample JavaScript code for testing."""
    return """
function calculateSum(a, b) {
    if (typeof a !== 'number' || typeof b !== 'number') {
        throw new Error('Arguments must be numbers');
    }
    return a + b;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return [...this.history];
    }
}

module.exports = { calculateSum, Calculator };
"""


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "language": "python",
        "test_framework": "pytest",
        "coverage_target": 85,
        "quality_target": 90,
        "security_rules": {
            "sql_injection": True,
            "xss_vulnerabilities": True,
            "authentication_bypass": True,
            "data_exposure": True
        },
        "test_patterns": {
            "edge_cases": True,
            "error_handling": True,
            "mocking": True,
            "integration_scenarios": False
        },
        "output": {
            "format": "standard",
            "include_docstrings": True,
            "add_comments": True
        }
    }


@pytest.fixture
def mock_security_scanner():
    """Provide a mock security scanner for testing."""
    mock_scanner = Mock()
    mock_scanner.scan.return_value = {
        "issues": [
            {
                "type": "sql_injection",
                "severity": "high",
                "line": 42,
                "description": "Potential SQL injection vulnerability",
                "code": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
                "recommendation": "Use parameterized queries"
            }
        ],
        "summary": {
            "total_issues": 1,
            "high_severity": 1,
            "medium_severity": 0,
            "low_severity": 0
        }
    }
    return mock_scanner


@pytest.fixture
def mock_coverage_analyzer():
    """Provide a mock coverage analyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.analyze.return_value = {
        "percentage": 85.7,
        "lines_covered": 120,
        "lines_total": 140,
        "missing_lines": [15, 23, 67, 89],
        "files": {
            "src/main.py": {
                "percentage": 90.0,
                "lines_covered": 45,
                "lines_total": 50
            },
            "src/utils.py": {
                "percentage": 80.0,
                "lines_covered": 75,
                "lines_total": 90
            }
        }
    }
    return mock_analyzer


class TestDataBuilder:
    """Builder class for creating test data."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder to initial state."""
        self._code = ""
        self._language = "python"
        self._test_framework = "pytest"
        self._config = {}
        return self
    
    def with_code(self, code: str):
        """Set code content."""
        self._code = code
        return self
    
    def with_language(self, language: str):
        """Set programming language."""
        self._language = language
        return self
    
    def with_test_framework(self, framework: str):
        """Set test framework."""
        self._test_framework = framework
        return self
    
    def with_config(self, config: Dict[str, Any]):
        """Set configuration."""
        self._config = config
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the test data."""
        return {
            "code": self._code,
            "language": self._language,
            "test_framework": self._test_framework,
            "config": self._config
        }


@pytest.fixture
def test_data_builder():
    """Provide a test data builder."""
    return TestDataBuilder()


def pytest_report_header(config):
    """Add custom header to pytest report."""
    return [
        f"TestGen-Copilot Test Suite",
        f"Test Environment: {os.environ.get('TESTGEN_ENV', 'unknown')}",
        f"Python Version: {sys.version.split()[0]}",
        f"Test Data Directory: {Path(__file__).parent / 'fixtures'}",
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom terminal summary."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "✅ All tests passed! TestGen-Copilot is ready for deployment.")
    else:
        terminalreporter.write_sep("=", "❌ Some tests failed. Please fix issues before deploying.")


# Custom assertions
def assert_valid_test_code(test_code: str):
    """Assert that generated test code is valid."""
    # Check for basic test structure
    assert "def test_" in test_code or "class Test" in test_code, "No test functions/classes found"
    assert "assert" in test_code, "No assertions found in test code"
    
    # Check for imports
    assert any(imp in test_code for imp in ["import", "from"]), "No imports found"
    
    # Try to compile the code
    try:
        compile(test_code, "<test>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated test code has syntax errors: {e}")


def assert_security_scan_valid(scan_result: Dict[str, Any]):
    """Assert that security scan result is valid."""
    assert "issues" in scan_result, "Security scan missing 'issues' key"
    assert "summary" in scan_result, "Security scan missing 'summary' key"
    assert isinstance(scan_result["issues"], list), "Issues must be a list"
    assert isinstance(scan_result["summary"], dict), "Summary must be a dict"


# Make custom assertions available globally
pytest.assert_valid_test_code = assert_valid_test_code
pytest.assert_security_scan_valid = assert_security_scan_valid