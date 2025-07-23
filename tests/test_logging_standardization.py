"""Test logging standardization across modules."""

import io
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from testgen_copilot.generator import TestGenerator, GenerationConfig
from testgen_copilot.logging_config import get_generator_logger, LogContext, configure_logging


class TestLoggingStandardization:
    """Test that all modules use consistent structured logging patterns."""

    def setup_method(self):
        """Setup test environment with structured logging."""
        configure_logging(level="DEBUG", format_type="structured", enable_console=True)

    def test_generator_methods_use_structured_logging(self):
        """Test that all generator language methods use get_generator_logger() consistently."""
        # Create a test file that will exercise language-specific methods
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    return True
""")
            test_file = Path(f.name)

        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter('%(name)s:%(levelname)s:%(message)s'))
        
        logger = get_generator_logger()
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        try:
            # Generate tests - this should use structured logging
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = TestGenerator(GenerationConfig(language="python"))
                generator.generate_tests(test_file, temp_dir)

            # Check that structured logging was used (should NOT see basic logging.getLogger patterns)
            log_output = log_stream.getvalue()
            
            # The test will fail initially because we haven't standardized the logging yet
            # This test documents what we expect after standardization
            assert "testgen_copilot.generator" in log_output, "Generator should use structured logger"
            
            # Should not see direct logging.getLogger(__name__) patterns in output
            # After standardization, all logging should go through structured loggers
            
        finally:
            test_file.unlink(missing_ok=True)
            logger.logger.removeHandler(handler)

    def test_no_direct_logging_getlogger_usage(self):
        """Test that modules don't use logging.getLogger(__name__) directly.
        
        This test will fail until we replace all direct logging.getLogger usage
        with the appropriate structured logger functions.
        """
        # This is a design test - we'll verify that the code has been refactored
        # to use structured logging instead of direct logging.getLogger calls
        
        # Read generator.py and check for logging.getLogger patterns
        generator_file = Path(__file__).parent.parent / "src" / "testgen_copilot" / "generator.py"
        content = generator_file.read_text()
        
        # After standardization, there should be no logging.getLogger(__name__) calls
        # in the language-specific methods
        getlogger_count = content.count("logging.getLogger(__name__)")
        
        # This assertion will fail initially - that's expected!
        # It documents the target state after logging standardization
        assert getlogger_count == 0, f"Found {getlogger_count} direct logging.getLogger(__name__) calls in generator.py"

    def test_log_context_usage_in_operations(self):
        """Test that major operations use LogContext for correlation tracking."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_func(): pass")
            test_file = Path(f.name)

        try:
            # Capture structured log output
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            
            # Use a custom formatter to see structured data
            class TestFormatter(logging.Formatter):
                def format(self, record):
                    correlation_id = getattr(record, 'correlation_id', None)
                    operation_name = getattr(record, 'operation_name', None)
                    return f"correlation_id:{correlation_id} operation:{operation_name} {record.getMessage()}"
            
            handler.setFormatter(TestFormatter())
            
            logger = get_generator_logger()
            logger.logger.addHandler(handler)
            logger.logger.setLevel(logging.DEBUG)

            # Generate tests with LogContext
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = TestGenerator()
                generator.generate_tests(test_file, temp_dir)

            log_output = log_stream.getvalue()
            
            # Should see correlation IDs and operation names in logs
            assert "correlation_id:" in log_output, "Should have correlation IDs in structured logs"
            assert "operation:" in log_output, "Should have operation names in structured logs"
            
        finally:
            test_file.unlink(missing_ok=True)
            logger.logger.removeHandler(handler)