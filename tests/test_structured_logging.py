"""Test structured logging implementation."""

import json
import logging
import tempfile
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, 'src')

from testgen_copilot.logging_config import StructuredLogger, configure_logging, LogContext, StructuredFormatter


class TestStructuredLogger:
    """Test structured logging functionality."""

    def test_structured_logger_basic_functionality(self):
        """Test basic structured logging functionality."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        # Test basic structured log
        logger.info("Test message", {"key": "value", "number": 42})
        
        log_output = output.getvalue()
        assert "Test message" in log_output
        assert "key" in log_output
        assert "value" in log_output

    def test_log_context_manager(self):
        """Test log context manager for operation tracking."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        with LogContext(logger, "test_operation", {"user_id": "123"}):
            logger.info("Operation in progress")
        
        log_output = output.getvalue()
        assert "test_operation" in log_output
        assert "user_id" in log_output
        assert "123" in log_output

    def test_performance_timing(self):
        """Test performance timing functionality."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        # Test timing context manager
        with logger.time_operation("slow_operation"):
            # Simulate some work
            pass
        
        log_output = output.getvalue()
        assert "slow_operation" in log_output
        assert "duration_ms" in log_output

    def test_error_logging_with_context(self):
        """Test error logging with exception context."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.ERROR)
        logger.logger.propagate = False
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Operation failed", {"error_type": "validation"}, exc_info=True)
        
        log_output = output.getvalue()
        assert "Operation failed" in log_output
        assert "error_type" in log_output
        assert "validation" in log_output

    def test_json_structured_format(self):
        """Test JSON structured logging format."""
        output = StringIO()
        
        # Create logger with JSON formatter
        logger = StructuredLogger("test.module", use_json=True)
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=True)
        handler.setFormatter(formatter)
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        logger.info("Test message", {"key": "value", "count": 10})
        
        log_output = output.getvalue().strip()
        
        # Should be valid JSON
        try:
            log_data = json.loads(log_output)
            assert log_data["message"] == "Test message"
            assert log_data["key"] == "value"
            assert log_data["count"] == 10
            assert "timestamp" in log_data
            assert "level" in log_data
            assert "module" in log_data
        except json.JSONDecodeError:
            assert False, f"Log output is not valid JSON: {log_output}"

    def test_configure_logging_function(self):
        """Test the centralized logging configuration function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            # Test configuration
            configure_logging(
                level="INFO",
                format_type="json",
                log_file=str(log_file),
                enable_console=False
            )
            
            # Create a logger and test it
            logger = StructuredLogger("test.configured")
            logger.info("Configuration test", {"test": True})
            
            # Check log file was created
            assert log_file.exists()
            
            # Check log content
            log_content = log_file.read_text()
            assert "Configuration test" in log_content
            assert "test" in log_content

    def test_security_context_logging(self):
        """Test logging with security context for audit trails."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("security.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        # Test security audit log
        logger.audit("File access attempt", {
            "user_id": "user123",
            "file_path": "/sensitive/data.txt",
            "action": "read",
            "result": "denied",
            "ip_address": "192.168.1.100"
        })
        
        log_output = output.getvalue()
        assert "File access attempt" in log_output
        assert "user123" in log_output
        assert "denied" in log_output
        assert "AUDIT" in log_output or "audit" in log_output

    def test_performance_metrics_collection(self):
        """Test performance metrics collection and logging."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("perf.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        # Test metrics logging
        logger.metrics("test_generation_complete", {
            "files_processed": 5,
            "tests_generated": 25,
            "duration_ms": 1500,
            "memory_usage_mb": 45.2,
            "success_rate": 100.0
        })
        
        log_output = output.getvalue()
        assert "test_generation_complete" in log_output
        assert "files_processed" in log_output
        assert "25" in log_output
        assert "METRICS" in log_output or "metrics" in log_output

    def test_correlation_id_tracking(self):
        """Test correlation ID tracking across operations."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        correlation_id = "req-123-456"
        
        with LogContext(logger, "user_request", {"correlation_id": correlation_id}):
            logger.info("Processing request")
            logger.info("Validating input")
            logger.info("Request completed")
        
        log_output = output.getvalue()
        lines = log_output.strip().split('\n')
        
        # All log lines should contain the correlation ID
        for line in lines:
            assert correlation_id in line

    def test_log_filtering_and_sampling(self):
        """Test log filtering and sampling for high-volume scenarios."""
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=False)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.module")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.DEBUG)
        logger.logger.propagate = False
        
        # Test that we can filter certain types of logs
        for i in range(10):
            if i % 3 == 0:  # Sample every 3rd log
                logger.debug(f"Debug message {i}", {"iteration": i, "sampled": True})
            else:
                logger.debug(f"Debug message {i}", {"iteration": i, "sampled": False})
        
        log_output = output.getvalue()
        
        # Should have fewer sampled logs
        sampled_count = log_output.count('"sampled": true') if '"sampled": true' in log_output else log_output.count('sampled": True')
        total_count = log_output.count("Debug message")
        
        assert sampled_count < total_count

    def test_datetime_timezone_aware(self):
        """Test that timestamp uses timezone-aware datetime instead of deprecated utcnow()."""
        import warnings
        import datetime
        
        output = StringIO()
        handler = logging.StreamHandler(output)
        formatter = StructuredFormatter(use_json=True)
        handler.setFormatter(formatter)
        
        logger = StructuredLogger("test.datetime")
        logger.logger.handlers = [handler]
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
        
        # Capture warnings to ensure no deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            logger.info("Testing timezone aware datetime")
            
            # Check that no deprecation warnings were raised
            datetime_warnings = [warning for warning in w 
                               if "datetime.datetime.utcnow" in str(warning.message)]
            assert len(datetime_warnings) == 0, f"Found deprecation warnings: {[str(w.message) for w in datetime_warnings]}"
        
        # Verify the timestamp format is correct
        log_output = output.getvalue()
        assert "timestamp" in log_output
        # Should contain Z suffix for UTC timezone
        assert "Z" in log_output


def main():
    """Run structured logging tests."""
    print("🧪 Testing Structured Logging Implementation")
    print("=" * 50)
    
    test_instance = TestStructuredLogger()
    
    test_methods = [
        test_instance.test_structured_logger_basic_functionality,
        test_instance.test_log_context_manager,
        test_instance.test_performance_timing,
        test_instance.test_error_logging_with_context,
        test_instance.test_json_structured_format,
        test_instance.test_configure_logging_function,
        test_instance.test_security_context_logging,
        test_instance.test_performance_metrics_collection,
        test_instance.test_correlation_id_tracking,
        test_instance.test_log_filtering_and_sampling,
        test_instance.test_datetime_timezone_aware,
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
        print("✅ All structured logging tests passed!")
    else:
        print(f"❌ {failed} tests failed")


if __name__ == "__main__":
    main()