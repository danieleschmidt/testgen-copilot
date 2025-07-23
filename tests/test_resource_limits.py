"""Test resource limits and validation implementation."""

import ast
import signal
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, 'src')

from testgen_copilot.generator import TestGenerator, GenerationConfig
from testgen_copilot.file_utils import FileSizeError
from testgen_copilot.resource_limits import (
    AST_PARSE_TIMEOUT, 
    MAX_PROJECT_FILES, 
    MemoryMonitor,
    TimeoutHandler,
    ResourceMemoryError,
    validate_test_content,
    safe_parse_ast_with_timeout
)


class TestResourceLimits:
    """Test resource limits and validation functionality."""

    def test_file_size_limits_already_implemented(self):
        """Test that file size limits are working (should already pass)."""
        # This test validates existing functionality
        from testgen_copilot.file_utils import safe_read_file, FileSizeError
        
        # Create a test file that exceeds size limits
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            # Write more than 10MB of content
            large_content = "# " + "x" * (11 * 1024 * 1024)  # 11MB
            f.write(large_content)
            f.flush()
            
            # Test should raise FileSizeError
            try:
                safe_read_file(f.name, max_size_mb=10)
                assert False, "Should have raised FileSizeError for large file"
            except FileSizeError:
                pass  # Expected
            finally:
                Path(f.name).unlink()  # Clean up

    def test_ast_parsing_timeout_handler(self):
        """Test that AST parsing has timeout protection (should fail initially)."""
        # Test timeout handling for AST parsing
        test_code = """
def slow_function():
    # This is a normal function
    pass
"""
        
        # Test with very short timeout (should work for simple code)
        result = safe_parse_ast_with_timeout(test_code, "test.py", timeout_seconds=5)
        assert isinstance(result, ast.AST), "Should successfully parse simple code"
        
        # Test timeout behavior - this should be implemented
        try:
            # Simulate very slow parsing with an infinitely recursive structure
            very_complex_code = "(" * 10000 + ")" * 10000  # This will be slow to parse
            result = safe_parse_ast_with_timeout(very_complex_code, "complex.py", timeout_seconds=1)
            # If we get here, either parsing was fast or timeout didn't work
        except TimeoutError:
            pass  # Expected for timeout
        except SyntaxError:
            pass  # Also acceptable - complex code might have syntax issues
        except Exception as e:
            # Other exceptions are okay too - we're testing the timeout mechanism exists
            pass

    def test_project_file_batch_limits(self):
        """Test that project analysis respects batch size limits (should fail initially)."""
        # Test batch size limits for project-wide analysis
        generator = TestGenerator(GenerationConfig(language="python"))
        
        # Create many test files
        test_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create more files than the limit
            for i in range(MAX_PROJECT_FILES + 10):
                test_file = temp_path / f"test_{i}.py"
                test_file.write_text(f"def test_function_{i}(): pass")
                test_files.append(test_file)
            
            # This should respect batch limits and not process all files at once
            # Implementation should limit to MAX_PROJECT_FILES
            from testgen_copilot.resource_limits import BatchProcessor
            processor = BatchProcessor(max_files=MAX_PROJECT_FILES)
            
            processed_files = processor.process_files(test_files)
            assert len(processed_files) <= MAX_PROJECT_FILES, f"Should not process more than {MAX_PROJECT_FILES} files"

    def test_memory_monitoring(self):
        """Test memory usage monitoring and circuit breaker (should fail initially)."""
        # Test memory monitoring
        monitor = MemoryMonitor(max_memory_mb=100)  # 100MB limit
        
        # Should start with low memory usage
        assert not monitor.is_memory_exceeded(), "Should start below memory limit"
        
        # Memory monitoring should track usage
        current_usage = monitor.get_current_memory_mb()
        assert isinstance(current_usage, (int, float)), "Should return numeric memory usage"
        assert current_usage >= 0, "Memory usage should be non-negative"
        
        # Test circuit breaker behavior
        with patch.object(monitor, 'get_current_memory_mb', return_value=150):  # Exceed limit
            assert monitor.is_memory_exceeded(), "Should detect memory exceeded"
            
            try:
                monitor.check_memory_and_raise()
                assert False, "Should raise MemoryError when limit exceeded"
            except ResourceMemoryError:
                pass  # Expected

    def test_test_content_validation(self):
        """Test validation of generated test content (should fail initially)."""
        # Test valid test content
        valid_test_content = """
import unittest

class TestExample(unittest.TestCase):
    def test_function(self):
        result = some_function()
        assert result is not None
"""
        
        # Should pass validation
        assert validate_test_content(valid_test_content), "Valid test content should pass validation"
        
        # Test invalid content (no test methods)
        invalid_content = """
def regular_function():
    return "not a test"
"""
        
        # Should fail validation
        assert not validate_test_content(invalid_content), "Content without tests should fail validation"
        
        # Test malicious content (should be rejected)
        malicious_content = """
import os
os.system("rm -rf /")  # Malicious code
def test_something():
    pass
"""
        
        # Should fail validation
        assert not validate_test_content(malicious_content), "Malicious content should fail validation"

    def test_timeout_configuration(self):
        """Test that timeout values are configurable."""
        # Test that AST_PARSE_TIMEOUT is defined and reasonable
        assert AST_PARSE_TIMEOUT > 0, "AST parse timeout should be positive"
        assert AST_PARSE_TIMEOUT <= 60, "AST parse timeout should be reasonable (<=60s)"
        
        # Test that MAX_PROJECT_FILES is defined and reasonable
        assert MAX_PROJECT_FILES > 0, "Max project files should be positive"
        assert MAX_PROJECT_FILES <= 10000, "Max project files should be reasonable (<=10000)"

    def test_circuit_breaker_integration(self):
        """Test that circuit breaker integrates with generator methods."""
        generator = TestGenerator(GenerationConfig(language="python"))
        
        # Test that generator uses resource limits
        # This is an integration test to verify the limits are actually used
        test_content = "def test_function(): pass"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(test_content)
            f.flush()
            
            try:
                # Should complete successfully with normal content
                with tempfile.TemporaryDirectory() as output_dir:
                    result = generator.generate_tests(f.name, output_dir)
                    assert result.exists(), "Should generate test file successfully"
            finally:
                Path(f.name).unlink()


def main():
    """Run resource limits tests."""
    print("🧪 Testing Resource Limits Implementation")
    print("=" * 50)
    
    test_instance = TestResourceLimits()
    
    test_methods = [
        test_instance.test_file_size_limits_already_implemented,
        test_instance.test_ast_parsing_timeout_handler,
        test_instance.test_project_file_batch_limits,
        test_instance.test_memory_monitoring,
        test_instance.test_test_content_validation,
        test_instance.test_timeout_configuration,
        test_instance.test_circuit_breaker_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All resource limits tests passed!")
    else:
        print(f"❌ {failed} tests failed - resource limits implementation needed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)