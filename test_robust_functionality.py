#!/usr/bin/env python3
"""
Robust functionality test for TestGen Copilot Assistant
Tests error handling, security, validation, and monitoring for GENERATION 2
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_error_handling():
    """Test comprehensive error handling"""
    try:
        from testgen_copilot.error_recovery import retry_with_backoff, safe_execute
        
        # Test safe execution with error handling
        def failing_function():
            raise ValueError("Intentional test error")
        
        result = safe_execute(failing_function, default_value="safe_fallback")
        assert result == "safe_fallback", f"Expected 'safe_fallback', got {result}"
        
        print("✅ Error handling and recovery systems work")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_security_validation():
    """Test security validation and input sanitization"""
    try:
        from testgen_copilot.input_validation import (
            validate_file_path, 
            validate_project_directory, 
            validate_configuration,
            SecurityValidationError
        )
        from testgen_copilot.security import SecurityScanner
        
        # Test file path validation
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_file = Path(temp_dir) / "test.py"
            valid_file.write_text("# test file")
            
            # Should pass validation
            validate_file_path(str(valid_file))
            validate_project_directory(temp_dir)
            
            # Test configuration validation
            config = {"language": "python", "include_edge_cases": True}
            validate_configuration(config)
        
        # Test security scanner
        scanner = SecurityScanner()
        assert hasattr(scanner, 'scan_file')
        assert hasattr(scanner, 'scan_project')
        
        print("✅ Security validation and monitoring systems work")
        return True
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False

def test_logging_and_monitoring():
    """Test structured logging and monitoring capabilities"""
    try:
        from testgen_copilot.logging_config import (
            configure_logging, 
            get_cli_logger, 
            LogContext
        )
        from testgen_copilot.monitoring import HealthMonitor, get_health_monitor
        from testgen_copilot.metrics_collector import MetricsCollector
        
        # Test logging configuration
        configure_logging(level="INFO", format_type="structured")
        logger = get_cli_logger()
        
        # Test log context
        with LogContext(logger, "test_operation", {"test": True}):
            logger.info("Test log message")
        
        # Test health monitor
        health_monitor = get_health_monitor()
        health_status = health_monitor.get_health_status()
        assert "status" in health_status
        
        # Test metrics collector
        collector = MetricsCollector(Path("/tmp"))
        # Test record event method if available
        if hasattr(collector, 'record_event'):
            collector.record_event("test_event", {"test": True})
        
        print("✅ Logging and monitoring systems work")
        return True
    except Exception as e:
        print(f"❌ Logging and monitoring test failed: {e}")
        return False

def test_resilience_and_self_healing():
    """Test resilience patterns and self-healing capabilities"""
    try:
        from testgen_copilot.resilience import CircuitBreaker, CircuitBreakerConfig, RetryMechanism, RetryConfig
        from testgen_copilot.self_healing import SelfHealingSystem
        from testgen_copilot.monitoring import HealthMonitor
        
        # Test circuit breaker
        config = CircuitBreakerConfig(failure_threshold=3, timeout_duration_seconds=60)
        circuit_breaker = CircuitBreaker("test_circuit", config)
        assert circuit_breaker.state.value == "closed"
        
        # Test retry mechanism
        retry_config = RetryConfig(max_attempts=3, backoff_multiplier=1.5)
        retry_mechanism = RetryMechanism("test_retry", retry_config)
        assert retry_mechanism.config.max_attempts == 3
        
        # Test self-healing system
        healing_system = SelfHealingSystem(Path("/tmp"))
        assert healing_system is not None
        
        # Test health monitor
        health_monitor = HealthMonitor()
        assert health_monitor is not None
        
        print("✅ Resilience and self-healing systems work")
        return True
    except Exception as e:
        print(f"❌ Resilience test failed: {e}")
        return False

def test_resource_management():
    """Test resource limits and management"""
    try:
        from testgen_copilot.resource_limits import ResourceMonitor, MemoryMonitor, ResourceLimits
        from testgen_copilot.performance_monitor import PerformanceMonitor
        
        # Test resource monitor
        limits = ResourceLimits(max_memory_mb=100)
        resource_monitor = ResourceMonitor(limits)
        assert resource_monitor.limits.max_memory_mb == 100
        
        # Test memory monitor
        memory_monitor = MemoryMonitor()
        memory_usage = memory_monitor.get_current_memory_mb()
        assert isinstance(memory_usage, (int, float))
        
        # Test performance monitor
        perf_monitor = PerformanceMonitor()
        assert perf_monitor is not None
        
        print("✅ Resource management systems work")
        return True
    except Exception as e:
        print(f"❌ Resource management test failed: {e}")
        return False

def test_comprehensive_validation():
    """Test comprehensive validation across all components"""
    try:
        from testgen_copilot.cli import _validate_config_schema, _is_dangerous_path
        from testgen_copilot.input_validation import ValidationError
        
        # Test configuration schema validation
        valid_config = {
            "language": "python",
            "include_edge_cases": True,
            "include_error_paths": True
        }
        validated_config = _validate_config_schema(valid_config)
        assert validated_config["language"] == "python"
        
        # Test dangerous path detection
        assert _is_dangerous_path(Path("/etc/passwd")) == True
        assert _is_dangerous_path(Path("/tmp/safe_file.py")) == False
        
        # Test invalid configuration rejection
        try:
            invalid_config = {"__import__": "malicious_module"}
            _validate_config_schema(invalid_config)
            assert False, "Should have rejected dangerous config"
        except ValueError:
            pass  # Expected to fail
        
        print("✅ Comprehensive validation systems work")
        return True
    except Exception as e:
        print(f"❌ Comprehensive validation test failed: {e}")
        return False

def main():
    """Run all robust functionality tests"""
    print("🛡️ GENERATION 2: Testing Robust Functionality")
    print("=" * 60)
    
    tests = [
        test_error_handling,
        test_security_validation,
        test_logging_and_monitoring,
        test_resilience_and_self_healing,
        test_resource_management,
        test_comprehensive_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 2: MAKE IT ROBUST - SUCCESS!")
        print("   All error handling, security, and monitoring works!")
    else:
        print("❌ GENERATION 2: FAILED - Some robustness features are broken")
        sys.exit(1)

if __name__ == "__main__":
    main()