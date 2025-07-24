"""Test cross-platform timeout handling implementation."""

import ast
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, 'src')

from testgen_copilot.resource_limits import (
    CrossPlatformTimeoutHandler, 
    safe_parse_ast_with_timeout
)


class TestCrossPlatformTimeout:
    """Test cross-platform timeout functionality."""

    def test_cross_platform_timeout_handler_basic_functionality(self):
        """Test basic timeout handler functionality works on all platforms."""
        timeout_seconds = 2
        
        # Test successful completion within timeout
        with CrossPlatformTimeoutHandler(timeout_seconds):
            time.sleep(0.1)  # Short operation should complete
            result = True
        
        assert result, "Should complete successfully within timeout"

    def test_cross_platform_timeout_handler_timeout_triggers(self):
        """Test that timeout actually triggers for long operations."""
        timeout_seconds = 1
        
        start_time = time.time()
        try:
            with CrossPlatformTimeoutHandler(timeout_seconds):
                time.sleep(2)  # Operation longer than timeout
                assert False, "Should have raised TimeoutError"
        except TimeoutError as e:
            end_time = time.time()
            actual_duration = end_time - start_time
            assert "timed out" in str(e).lower(), "Error message should mention timeout"
            # Allow some tolerance for timing
            assert 0.8 <= actual_duration <= 1.5, f"Should timeout around {timeout_seconds}s, got {actual_duration}s"

    def test_cross_platform_timeout_handler_cleanup(self):
        """Test that timeout handler properly cleans up resources."""
        timeout_seconds = 1
        
        # Check that we can use multiple timeout handlers sequentially
        for i in range(3):
            with CrossPlatformTimeoutHandler(timeout_seconds):
                time.sleep(0.1)  # Short operation
                pass
        
        # Should complete without issues - tests cleanup

    def test_cross_platform_timeout_with_ast_parsing(self):
        """Test timeout integration with AST parsing operations."""
        # Simple valid code should parse quickly
        simple_code = "def simple(): pass"
        
        tree = safe_parse_ast_with_timeout(simple_code, "test.py", timeout_seconds=5)
        assert isinstance(tree, ast.AST), "Should parse successfully with timeout"
        
        # Verify we can extract functions from the parsed AST
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1, "Should find one function"
        assert functions[0].name == "simple", "Should parse function name correctly"

    def test_cross_platform_timeout_error_details(self):
        """Test that timeout errors provide helpful context."""
        timeout_seconds = 1
        
        try:
            with CrossPlatformTimeoutHandler(timeout_seconds):
                time.sleep(2)
                assert False, "Should have raised TimeoutError"
        except TimeoutError as e:
            error_msg = str(e)
            assert str(timeout_seconds) in error_msg, "Error should include timeout duration"
            assert "timed out" in error_msg.lower(), "Error should mention timeout"

    def test_cross_platform_timeout_works_on_windows(self):
        """Test that timeout works even when signal.SIGALRM is not available."""
        # Mock signal module to simulate Windows environment
        with patch('testgen_copilot.resource_limits.signal') as mock_signal:
            # Remove SIGALRM from the mock signal module to simulate Windows
            mock_signal.SIGALRM = None
            del mock_signal.SIGALRM
            
            # Should still work with threading-based timeout
            timeout_seconds = 1
            start_time = time.time()
            
            try:
                with CrossPlatformTimeoutHandler(timeout_seconds):
                    time.sleep(2)  # Operation longer than timeout
                    assert False, "Should have raised TimeoutError"
            except TimeoutError:
                end_time = time.time()
                actual_duration = end_time - start_time
                # Allow some tolerance for timing
                assert 0.8 <= actual_duration <= 1.5, f"Should timeout around {timeout_seconds}s even without signals"

    def test_cross_platform_timeout_threading_safety(self):
        """Test that timeout handler is thread-safe."""
        timeout_seconds = 2
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                with CrossPlatformTimeoutHandler(timeout_seconds):
                    time.sleep(0.5)  # Short operation
                    results.append(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.append(f"worker_{worker_id}_error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(results) == 3, f"All 3 workers should complete, got {len(results)}"
        assert len(errors) == 0, f"No errors should occur, got {errors}"

    def test_cross_platform_timeout_preserves_original_exception(self):
        """Test that original exceptions are preserved when no timeout occurs."""
        timeout_seconds = 2
        
        try:
            with CrossPlatformTimeoutHandler(timeout_seconds):
                # Raise a different exception before timeout
                raise ValueError("Original exception")
        except TimeoutError:
            assert False, "Should not have raised TimeoutError"
        except ValueError as e:
            assert str(e) == "Original exception", "Should preserve original exception"

    def test_cross_platform_timeout_context_manager_protocol(self):
        """Test that the timeout handler properly implements context manager protocol."""
        timeout_handler = CrossPlatformTimeoutHandler(1)
        
        # Test __enter__ returns self
        result = timeout_handler.__enter__()
        assert result is timeout_handler, "__enter__ should return self"
        
        # Test __exit__ can be called safely
        timeout_handler.__exit__(None, None, None)


def main():
    """Run cross-platform timeout tests."""
    print("🧪 Testing Cross-Platform Timeout Implementation")
    print("=" * 50)
    
    test_instance = TestCrossPlatformTimeout()
    
    test_methods = [
        test_instance.test_cross_platform_timeout_handler_basic_functionality,
        test_instance.test_cross_platform_timeout_handler_timeout_triggers,
        test_instance.test_cross_platform_timeout_handler_cleanup,
        test_instance.test_cross_platform_timeout_with_ast_parsing,
        test_instance.test_cross_platform_timeout_error_details,
        test_instance.test_cross_platform_timeout_works_on_windows,
        test_instance.test_cross_platform_timeout_threading_safety,
        test_instance.test_cross_platform_timeout_preserves_original_exception,
        test_instance.test_cross_platform_timeout_context_manager_protocol,
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
        print("✅ All cross-platform timeout tests passed!")
    else:
        print(f"❌ {failed} tests failed - cross-platform timeout implementation needed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)