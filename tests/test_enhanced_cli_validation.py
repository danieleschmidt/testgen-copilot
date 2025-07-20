"""Test enhanced CLI input validation features."""

import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testgen_copilot.cli import (
    _validate_config_schema, 
    _validate_paths, 
    _validate_numeric_args,
    _is_dangerous_path,
    _build_parser
)


class TestEnhancedCLIValidation:
    """Test enhanced CLI validation features."""

    def test_dangerous_config_keys_rejected(self):
        """Test that dangerous config keys are rejected."""
        dangerous_configs = [
            {"language": "python", "__import__": "os"},
            {"language": "python", "eval": "malicious_code"},
            {"language": "python", "exec": "dangerous"},
            {"language": "python", "open": "/etc/passwd"},
        ]
        
        for config in dangerous_configs:
            try:
                _validate_config_schema(config)
                assert False, f"Should have rejected dangerous config: {config}"
            except ValueError as e:
                assert "Dangerous config option detected" in str(e)

    def test_invalid_language_rejected(self):
        """Test that invalid languages are rejected."""
        invalid_config = {"language": "invalid_language"}
        
        try:
            _validate_config_schema(invalid_config)
            assert False, "Should have rejected invalid language"
        except ValueError as e:
            assert "Unsupported language" in str(e)

    def test_valid_languages_accepted(self):
        """Test that all valid languages are accepted."""
        valid_languages = ["python", "py", "javascript", "js", "typescript", "ts", "java", "c#", "csharp", "go", "rust"]
        
        for lang in valid_languages:
            config = {"language": lang}
            result = _validate_config_schema(config)
            assert result == config

    def test_dangerous_path_detection(self):
        """Test detection of dangerous system paths."""
        dangerous_paths = [
            Path("/etc/passwd"),
            Path("/proc/version"), 
            Path("/sys/kernel"),
            Path("/dev/null"),
            Path("/boot/config"),
            Path("/root/.ssh"),
            Path("/var/log/messages"),
            Path("/usr/bin/python"),
            Path("../../../etc/passwd"),
            Path("..\\..\\..\\windows\\system32"),
        ]
        
        for path in dangerous_paths:
            assert _is_dangerous_path(path), f"Should detect {path} as dangerous"

    def test_safe_path_detection(self):
        """Test that safe paths are not flagged as dangerous."""
        safe_paths = [
            Path("/home/user/project"),
            Path("/tmp/testgen"),
            Path("/opt/myapp"),
            Path("./local/path"),
            Path("project/src"),
            Path("/workspace/code"),
        ]
        
        for path in safe_paths:
            assert not _is_dangerous_path(path), f"Should not flag {path} as dangerous"

    def test_coverage_target_validation(self):
        """Test that coverage targets are validated."""
        parser = _build_parser()
        
        # Test valid coverage targets
        valid_targets = [0, 50, 85, 100]
        for target in valid_targets:
            args = parser.parse_args(["analyze", "--project", ".", "--coverage-target", str(target)])
            try:
                _validate_numeric_args(args, parser)
                # Should not raise exception for valid targets
            except SystemExit:
                assert False, f"Should accept valid coverage target {target}"
        
        # Test invalid coverage targets  
        invalid_targets = [-10, 150, -1, 101]
        for target in invalid_targets:
            args = parser.parse_args(["analyze", "--project", ".", "--coverage-target", str(target)])
            try:
                _validate_numeric_args(args, parser)
                assert False, f"Should reject invalid coverage target {target}"
            except SystemExit:
                pass  # Expected

    def test_quality_target_validation(self):
        """Test that quality targets are validated."""
        parser = _build_parser()
        
        # Test valid quality targets
        valid_targets = [0, 50, 90, 100]
        for target in valid_targets:
            args = parser.parse_args(["analyze", "--project", ".", "--quality-target", str(target)])
            try:
                _validate_numeric_args(args, parser)
                # Should not raise exception for valid targets
            except SystemExit:
                assert False, f"Should accept valid quality target {target}"
        
        # Test invalid quality targets
        invalid_targets = [-5, 150, -1, 105]
        for target in invalid_targets:
            args = parser.parse_args(["analyze", "--project", ".", "--quality-target", str(target)])
            try:
                _validate_numeric_args(args, parser)
                assert False, f"Should reject invalid quality target {target}"
            except SystemExit:
                pass  # Expected

    def test_poll_interval_validation(self):
        """Test that poll intervals are validated."""
        parser = _build_parser()
        
        # Test valid poll intervals
        valid_intervals = [0.1, 1.0, 5.0, 60.0, 300.0]
        for interval in valid_intervals:
            args = parser.parse_args(["generate", "--watch", ".", "--output", "tests", "--poll", str(interval)])
            try:
                _validate_numeric_args(args, parser)
                # Should not raise exception for valid intervals
            except SystemExit:
                assert False, f"Should accept valid poll interval {interval}"
        
        # Test invalid poll intervals
        invalid_intervals = [0, -1, 301, 1000]
        for interval in invalid_intervals:
            args = parser.parse_args(["generate", "--watch", ".", "--output", "tests", "--poll", str(interval)])
            try:
                _validate_numeric_args(args, parser)
                assert False, f"Should reject invalid poll interval {interval}"
            except SystemExit:
                pass  # Expected

    def test_path_validation_with_dangerous_paths(self):
        """Test that dangerous paths are rejected in path validation."""
        parser = _build_parser()
        
        # These should be rejected by the dangerous path check
        with tempfile.NamedTemporaryFile(mode='w', dir='/tmp', delete=False) as f:
            f.write("def test(): pass")
            safe_file = Path(f.name)
        
        try:
            # This should work - safe path
            args = parser.parse_args(["generate", "--file", str(safe_file), "--output", "/tmp/safe_output"])
            _validate_paths(args, parser)
            
            # Clean up
            safe_file.unlink()
            
        except SystemExit:
            # Clean up on failure
            if safe_file.exists():
                safe_file.unlink()
            assert False, "Should accept safe file path"

    def test_improved_error_messages(self):
        """Test that error messages are more descriptive."""
        # Test type error messages
        invalid_config = {
            "language": "python",
            "include_edge_cases": "yes"  # Should be bool
        }
        
        try:
            _validate_config_schema(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "expected bool" in error_msg
            assert "got str" in error_msg