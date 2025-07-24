"""Test standardized logging patterns in generator module."""

import ast
import logging
import re
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for testing  
sys.path.insert(0, 'src')

from testgen_copilot.generator import TestGenerator, GenerationConfig
from testgen_copilot.logging_config import get_generator_logger, StructuredFormatter


class TestGeneratorLoggingStandards:
    """Test that generator module uses standardized logging patterns."""

    def test_all_language_methods_use_structured_logging(self):
        """Test that all language generation methods use get_generator_logger() instead of logging.getLogger(__name__)."""
        
        # Read the generator.py source to check for logging patterns
        generator_path = Path("src/testgen_copilot/generator.py")
        content = generator_path.read_text()
        
        # Should NOT find any instances of logging.getLogger(__name__)
        legacy_logger_pattern = r'logging\.getLogger\(__name__\)'
        legacy_matches = re.findall(legacy_logger_pattern, content)
        
        assert len(legacy_matches) == 0, f"Found {len(legacy_matches)} instances of legacy logging.getLogger(__name__) pattern: {legacy_matches}"
        
        # Should find multiple instances of get_generator_logger()
        structured_logger_pattern = r'get_generator_logger\(\)'
        structured_matches = re.findall(structured_logger_pattern, content)
        
        # We expect at least one use in each language method
        expected_min_uses = 6  # Python, JS, Java, C#, Go, Rust methods should all use it
        assert len(structured_matches) >= expected_min_uses, f"Expected at least {expected_min_uses} uses of get_generator_logger(), found {len(structured_matches)}"

    def test_javascript_generation_uses_structured_logging(self):
        """Test that JavaScript generation methods use structured logging with context."""
        # Check that the JavaScript generation method uses get_generator_logger()
        generator_path = Path("src/testgen_copilot/generator.py")
        content = generator_path.read_text()
        
        # Check that _generate_javascript_tests uses structured logging
        js_method_start = content.find("def _generate_javascript_tests(")
        js_method_end = content.find("def _parse_js_functions(", js_method_start)
        js_method_content = content[js_method_start:js_method_end]
        
        assert "get_generator_logger()" in js_method_content, "JavaScript generation should use get_generator_logger()"
        assert "logger.info" in js_method_content or "logger.debug" in js_method_content, "Should use structured logging methods"
        assert "time_operation" in js_method_content, "Should use performance timing"

    def test_all_language_methods_have_error_logging(self):
        """Test that all language generation methods have proper error logging with structured context."""
        # Check that all language methods have proper error logging
        generator_path = Path("src/testgen_copilot/generator.py")
        content = generator_path.read_text()
        
        # All language generation methods should have error logging
        language_methods = [
            "_generate_javascript_tests",
            "_generate_java_tests", 
            "_generate_csharp_tests",
            "_generate_go_tests",
            "_generate_rust_tests"
        ]
        
        for method_name in language_methods:
            method_start = content.find(f"def {method_name}(")
            if method_start == -1:
                continue  # Skip if method not found
                
            # Find the end of the method (next def or end of class)
            next_method = content.find("\n    def ", method_start + 1)
            if next_method == -1:
                method_content = content[method_start:]
            else:
                method_content = content[method_start:next_method]
            
            assert "logger.error" in method_content, f"{method_name} should have error logging"
            assert "get_generator_logger()" in method_content, f"{method_name} should use structured logger"

    def test_go_and_rust_methods_have_logging(self):
        """Test that Go and Rust generation methods have logging."""
        # Check that Go and Rust generation methods use structured logging
        generator_path = Path("src/testgen_copilot/generator.py")
        content = generator_path.read_text()
        
        # Check Go methods
        go_method_start = content.find("def _generate_go_tests(")
        go_method_end = content.find("def _parse_go_functions(", go_method_start)
        go_method_content = content[go_method_start:go_method_end]
        
        assert "get_generator_logger()" in go_method_content, "Go generation should use get_generator_logger()"
        assert "logger.info" in go_method_content or "logger.debug" in go_method_content, "Go generation should use structured logging"
        assert "time_operation" in go_method_content, "Go generation should use performance timing"
        
        # Check Rust methods
        rust_method_start = content.find("def _generate_rust_tests(")
        rust_method_end = content.find("def _parse_rust_functions(", rust_method_start)
        rust_method_content = content[rust_method_start:rust_method_end]
        
        assert "get_generator_logger()" in rust_method_content, "Rust generation should use get_generator_logger()"
        assert "logger.info" in rust_method_content or "logger.debug" in rust_method_content, "Rust generation should use structured logging"
        assert "time_operation" in rust_method_content, "Rust generation should use performance timing"

    def test_log_context_usage_in_major_operations(self):
        """Test that major operations use LogContext for correlation tracking."""
        # Read the generator.py source to check for LogContext usage
        generator_path = Path("src/testgen_copilot/generator.py")
        content = generator_path.read_text()
        
        # Should find LogContext usage in major operations
        log_context_pattern = r'with LogContext\('
        context_matches = re.findall(log_context_pattern, content)
        
        # We expect LogContext to be used in the main generate_tests method
        assert len(context_matches) >= 1, f"Expected LogContext usage in major operations, found {len(context_matches)}"
        
        # Should also find timing operations
        timing_pattern = r'with logger\.time_operation\('
        timing_matches = re.findall(timing_pattern, content)
        
        assert len(timing_matches) >= 2, f"Expected timing operations in language-specific methods, found {len(timing_matches)}"


def main():
    """Run generator logging standardization tests."""
    print("🧪 Testing Generator Logging Standards")
    print("=" * 50)
    
    test_instance = TestGeneratorLoggingStandards()
    
    test_methods = [
        test_instance.test_all_language_methods_use_structured_logging,
        test_instance.test_javascript_generation_uses_structured_logging,
        test_instance.test_all_language_methods_have_error_logging,
        test_instance.test_go_and_rust_methods_have_logging,
        test_instance.test_log_context_usage_in_major_operations,
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
        print("✅ All generator logging tests passed!")
    else:
        print(f"❌ {failed} tests failed - logging standardization needed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)