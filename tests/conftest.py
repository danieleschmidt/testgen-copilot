"""Shared pytest configuration and fixtures for TestGen Copilot Assistant."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import Mock, patch

import pytest

# Ensure 'src' directory is on sys.path so tests can import the package without installation
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import modules after path setup
try:
    from testgen_copilot.core import TestGenerator
    from testgen_copilot.cli import CLI
    from testgen_copilot.quality import QualityScorer
except ImportError:
    # Modules may not exist yet during initial setup
    TestGenerator = None
    CLI = None
    QualityScorer = None


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that take longer than 1 second
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_code_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    code_content = '''
def calculate_discount(price, discount_percent):
    """Calculate discounted price."""
    return price * (1 - discount_percent / 100)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''
    file_path = temp_dir / "sample.py"
    file_path.write_text(code_content)
    return file_path


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Get the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Load test data from JSON fixture."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    test_data_file = fixtures_dir / "test_data.json"
    
    if test_data_file.exists():
        return json.loads(test_data_file.read_text())
    return {}


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Default configuration for TestGen Copilot."""
    return {
        "language": "python",
        "test_framework": "pytest",
        "coverage_target": 85,
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
def config_file(temp_dir: Path, default_config: Dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    config_path = temp_dir / ".testgen.config.json"
    config_path.write_text(json.dumps(default_config, indent=2))
    return config_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        "choices": [{
            "message": {
                "content": """
def test_calculate_discount():
    assert calculate_discount(100, 10) == 90.0
    assert calculate_discount(100, 0) == 100.0
    assert calculate_discount(100, 100) == 0.0
"""
            }
        }]
    }


@pytest.fixture
def mock_security_scan_result():
    """Mock security scan result."""
    return {
        "issues": [
            {
                "type": "sql_injection",
                "severity": "high",
                "line": 10,
                "description": "Potential SQL injection vulnerability",
                "suggestion": "Use parameterized queries"
            }
        ],
        "score": 75
    }


@pytest.fixture
def mock_coverage_result():
    """Mock coverage analysis result."""
    return {
        "line_coverage": 85.5,
        "branch_coverage": 78.2,
        "function_coverage": 92.1,
        "missing_lines": [15, 23, 45],
        "total_lines": 100,
        "covered_lines": 85
    }


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def test_generator(default_config: Dict[str, Any]) -> TestGenerator:
    """Create a TestGenerator instance for testing."""
    if TestGenerator is None:
        pytest.skip("TestGenerator not available")
    return TestGenerator(config=default_config)


@pytest.fixture
def quality_scorer() -> QualityScorer:
    """Create a QualityScorer instance for testing."""
    if QualityScorer is None:
        pytest.skip("QualityScorer not available")
    return QualityScorer()


@pytest.fixture
def cli_runner() -> CLI:
    """Create a CLI instance for testing."""
    if CLI is None:
        pytest.skip("CLI not available")
    return CLI()


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def clean_environment():
    """Clean environment variables for testing."""
    original_env = os.environ.copy()
    # Clear TestGen-specific environment variables
    testgen_vars = [key for key in os.environ if key.startswith('TESTGEN_')]
    for var in testgen_vars:
        del os.environ[var]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        'TESTGEN_ENV': 'test',
        'TESTGEN_LOG_LEVEL': 'DEBUG',
        'TESTGEN_DATA_DIR': '/tmp/testgen',
        'OPENAI_API_KEY': 'test-key-123',
        'ANTHROPIC_API_KEY': 'test-key-456'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


# =============================================================================
# Performance Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts after each test."""
    yield
    
    # Clean up any temporary files or directories created during tests
    import shutil
    test_dirs = [
        ".testgen",
        "testgen.log",
        "profile.prof"
    ]
    
    for directory in test_dirs:
        if os.path.exists(directory):
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            else:
                os.remove(directory)
