"""Testing utilities and helper functions."""

import asyncio
import tempfile
import contextlib
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock
import pytest


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, response_content: str = "# Generated test code\ndef test_example():\n    assert True"):
        self.response_content = response_content
        self.call_count = 0
        self.last_request = None
    
    def chat_completion_create(self, **kwargs) -> Mock:
        """Mock chat completion."""
        self.call_count += 1
        self.last_request = kwargs
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = self.response_content
        return mock_response
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_request = None


class TestFileManager:
    """Manages temporary test files and directories."""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
    
    @contextlib.contextmanager
    def temp_directory(self) -> Generator[Path, None, None]:
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.temp_dirs.append(temp_path)
            try:
                yield temp_path
            finally:
                if temp_path in self.temp_dirs:
                    self.temp_dirs.remove(temp_path)
    
    @contextlib.contextmanager
    def temp_file(self, content: str = "", suffix: str = ".py") -> Generator[Path, None, None]:
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
            self.temp_files.append(temp_path)
        
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
        # Directories are handled by context managers


def create_sample_python_file(content: str, file_path: Path) -> Path:
    """Create a sample Python file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def create_sample_project_structure(base_dir: Path) -> Dict[str, Path]:
    """Create a realistic project structure for testing."""
    structure = {
        'src_dir': base_dir / 'src',
        'tests_dir': base_dir / 'tests',
        'docs_dir': base_dir / 'docs',
        'config_file': base_dir / '.testgen.config.json',
        'requirements_file': base_dir / 'requirements.txt',
        'readme_file': base_dir / 'README.md',
    }
    
    # Create directories
    for dir_path in [structure['src_dir'], structure['tests_dir'], structure['docs_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    structure['config_file'].write_text('{"language": "python", "test_framework": "pytest"}')
    structure['requirements_file'].write_text('pytest>=7.0.0\npytest-cov>=4.0.0\n')
    structure['readme_file'].write_text('# Test Project\n\nThis is a sample project for testing.')
    
    # Create sample source files
    main_py = structure['src_dir'] / 'main.py'
    main_py.write_text('''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers."""
    return a * b
'''.strip())
    
    utils_py = structure['src_dir'] / 'utils.py'
    utils_py.write_text('''
def validate_input(value: str) -> bool:
    """Validate input string."""
    return isinstance(value, str) and len(value.strip()) > 0

def format_output(data: dict) -> str:
    """Format dictionary as string."""
    return str(data)
'''.strip())
    
    structure['main_py'] = main_py
    structure['utils_py'] = utils_py
    
    return structure


async def async_test_helper(coro):
    """Helper for running async functions in tests."""
    return await coro


def parametrize_test_data():
    """Common test data for parametrized tests."""
    return [
        # (input, expected_output, description)
        ("def add(a, b): return a + b", "test_add", "Simple function"),
        ("class Calculator:\n    def add(self, a, b): return a + b", "TestCalculator", "Class method"),
        ("async def fetch_data(): return 'data'", "test_fetch_data", "Async function"),
    ]


class AsyncMockLLMClient:
    """Async mock LLM client for testing async operations."""
    
    def __init__(self, response_content: str = "# Generated test code\ndef test_example():\n    assert True"):
        self.response_content = response_content
        self.call_count = 0
        self.last_request = None
    
    async def achat_completion_create(self, **kwargs) -> Mock:
        """Mock async chat completion."""
        await asyncio.sleep(0.01)  # Simulate async delay
        self.call_count += 1
        self.last_request = kwargs
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = self.response_content
        return mock_response


def assert_test_file_valid(test_content: str, expected_patterns: Optional[list] = None):
    """Assert that generated test file contains expected patterns."""
    if expected_patterns is None:
        expected_patterns = [
            "import pytest",
            "def test_",
            "assert",
        ]
    
    for pattern in expected_patterns:
        assert pattern in test_content, f"Expected pattern '{pattern}' not found in test content"


def create_mock_security_scan_result():
    """Create a mock security scan result for testing."""
    return {
        "issues": [
            {
                "type": "sql_injection",
                "severity": "high",
                "line": 42,
                "description": "Potential SQL injection vulnerability",
                "code": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
            },
            {
                "type": "hardcoded_secret",
                "severity": "medium",
                "line": 15,
                "description": "Hardcoded API key detected",
                "code": "API_KEY = 'sk-1234567890abcdef'"
            }
        ],
        "summary": {
            "total_issues": 2,
            "high_severity": 1,
            "medium_severity": 1,
            "low_severity": 0
        }
    }


def performance_timer():
    """Context manager for measuring test performance."""
    import time
    
    @contextlib.contextmanager
    def timer():
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.3f} seconds")
    
    return timer()