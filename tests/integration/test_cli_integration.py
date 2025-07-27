"""
Integration tests for CLI functionality.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner

from testgen_copilot.cli import main


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create source directory
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    
    # Create sample Python file
    sample_file = src_dir / "calculator.py"
    sample_file.write_text('''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract second number from first."""
    return a - b

class Calculator:
    """A simple calculator."""
    
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        if operation == "add":
            result = add(a, b)
        elif operation == "subtract":
            result = subtract(a, b)
        else:
            raise ValueError("Unknown operation")
        
        self.history.append((operation, a, b, result))
        return result
''')
    
    # Create tests directory
    tests_dir = temp_dir / "tests"
    tests_dir.mkdir()
    
    # Create config file
    config_file = temp_dir / ".testgen.config.json"
    config_file.write_text('''{
    "language": "python",
    "test_framework": "pytest",
    "coverage_target": 80,
    "security_rules": {
        "sql_injection": true,
        "xss_vulnerabilities": false
    },
    "test_patterns": {
        "edge_cases": true,
        "error_handling": true,
        "mocking": true
    }
}''')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.integration
def test_generate_command_single_file(temp_project_dir):
    """Test generating tests for a single file."""
    runner = CliRunner()
    
    src_file = temp_project_dir / "src" / "calculator.py"
    output_dir = temp_project_dir / "tests"
    
    result = runner.invoke(main, [
        'generate',
        '--file', str(src_file),
        '--output', str(output_dir)
    ])
    
    assert result.exit_code == 0
    assert "Generated tests" in result.output
    
    # Check that test file was created
    test_files = list(output_dir.glob("test_*.py"))
    assert len(test_files) > 0
    
    # Check test file content
    test_file = test_files[0]
    content = test_file.read_text()
    assert "def test_" in content
    assert "Calculator" in content or "add" in content


@pytest.mark.integration
def test_generate_command_with_config(temp_project_dir):
    """Test generating tests using configuration file."""
    runner = CliRunner()
    
    src_file = temp_project_dir / "src" / "calculator.py"
    output_dir = temp_project_dir / "tests"
    
    # Change to project directory so config is picked up
    result = runner.invoke(main, [
        'generate',
        '--file', str(src_file),
        '--output', str(output_dir)
    ], cwd=temp_project_dir)
    
    assert result.exit_code == 0
    
    # Verify config was used
    test_files = list(output_dir.glob("test_*.py"))
    assert len(test_files) > 0


@pytest.mark.integration
def test_analyze_command(temp_project_dir):
    """Test the analyze command."""
    runner = CliRunner()
    
    result = runner.invoke(main, [
        'analyze',
        '--project', str(temp_project_dir),
        '--coverage-target', '80'
    ])
    
    assert result.exit_code == 0
    assert "Coverage analysis" in result.output or "Analysis complete" in result.output


@pytest.mark.integration
def test_generate_project_mode(temp_project_dir):
    """Test generating tests for entire project."""
    runner = CliRunner()
    
    output_dir = temp_project_dir / "tests"
    
    result = runner.invoke(main, [
        'generate',
        '--project', str(temp_project_dir),
        '--output', str(output_dir),
        '--batch'
    ])
    
    assert result.exit_code == 0
    
    # Check that tests were generated
    test_files = list(output_dir.glob("test_*.py"))
    assert len(test_files) > 0


@pytest.mark.integration
def test_security_scan_integration(temp_project_dir):
    """Test security scanning integration."""
    runner = CliRunner()
    
    # Create a file with potential security issues
    vulnerable_file = temp_project_dir / "src" / "vulnerable.py"
    vulnerable_file.write_text('''
import os

def execute_command(user_input):
    # Potential command injection
    os.system(f"echo {user_input}")

def sql_query(user_id):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query
''')
    
    result = runner.invoke(main, [
        'generate',
        '--file', str(vulnerable_file),
        '--output', str(temp_project_dir / "tests"),
        '--security-scan'
    ])
    
    # Should complete but may warn about security issues
    assert result.exit_code in [0, 1]  # May exit with 1 due to security findings
    
    if result.exit_code == 1:
        assert "security" in result.output.lower() or "vulnerability" in result.output.lower()


@pytest.mark.integration
def test_coverage_integration(temp_project_dir):
    """Test coverage analysis integration."""
    runner = CliRunner()
    
    # First generate tests
    src_file = temp_project_dir / "src" / "calculator.py"
    output_dir = temp_project_dir / "tests"
    
    result = runner.invoke(main, [
        'generate',
        '--file', str(src_file),
        '--output', str(output_dir)
    ])
    
    assert result.exit_code == 0
    
    # Then analyze coverage
    result = runner.invoke(main, [
        'analyze',
        '--project', str(temp_project_dir),
        '--coverage-target', '70',
        '--show-missing'
    ])
    
    assert result.exit_code == 0


@pytest.mark.integration
def test_watch_mode_basic(temp_project_dir):
    """Test basic watch mode functionality (short duration)."""
    import threading
    import time
    
    runner = CliRunner()
    src_dir = temp_project_dir / "src"
    output_dir = temp_project_dir / "tests"
    
    # Run watch mode in a separate thread for a short time
    watch_result = None
    watch_exception = None
    
    def run_watch():
        nonlocal watch_result, watch_exception
        try:
            watch_result = runner.invoke(main, [
                'generate',
                '--watch', str(src_dir),
                '--output', str(output_dir),
                '--poll', '0.5'
            ], input='\n')  # Send newline to stop watch mode
        except Exception as e:
            watch_exception = e
    
    watch_thread = threading.Thread(target=run_watch)
    watch_thread.start()
    
    # Let it run briefly
    time.sleep(1)
    
    # Create a new file to trigger watch
    new_file = src_dir / "new_module.py"
    new_file.write_text('''
def hello_world():
    return "Hello, World!"
''')
    
    # Wait a bit more then stop
    time.sleep(1)
    
    # The thread should complete (either normally or with interrupt)
    watch_thread.join(timeout=5)
    
    if watch_exception:
        # Watch mode might be interrupted, which is expected
        assert "KeyboardInterrupt" in str(type(watch_exception))
    
    # Clean up
    if new_file.exists():
        new_file.unlink()


@pytest.mark.integration
def test_error_handling_invalid_file(temp_project_dir):
    """Test error handling for invalid files."""
    runner = CliRunner()
    
    invalid_file = temp_project_dir / "nonexistent.py"
    output_dir = temp_project_dir / "tests"
    
    result = runner.invoke(main, [
        'generate',
        '--file', str(invalid_file),
        '--output', str(output_dir)
    ])
    
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()


@pytest.mark.integration
def test_verbose_logging(temp_project_dir):
    """Test verbose logging output."""
    runner = CliRunner()
    
    src_file = temp_project_dir / "src" / "calculator.py"
    output_dir = temp_project_dir / "tests"
    
    result = runner.invoke(main, [
        '--log-level', 'debug',
        'generate',
        '--file', str(src_file),
        '--output', str(output_dir)
    ])
    
    assert result.exit_code == 0
    # Debug logging should produce more output
    assert len(result.output) > 100


@pytest.mark.integration
def test_quality_target_enforcement(temp_project_dir):
    """Test quality target enforcement."""
    runner = CliRunner()
    
    result = runner.invoke(main, [
        'analyze',
        '--project', str(temp_project_dir),
        '--quality-target', '95'  # Very high target
    ])
    
    # May pass or fail depending on quality, but should handle gracefully
    assert result.exit_code in [0, 1]
    assert "quality" in result.output.lower()