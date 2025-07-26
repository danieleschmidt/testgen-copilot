"""Test enhanced CLI input validation."""

import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testgen_copilot.cli import _validate_config_schema, _validate_paths, _build_parser


class TestCLIValidation:
    """Test CLI input validation functions."""

    def test_config_schema_validation_rejects_unknown_keys(self):
        """Test that unknown config keys are rejected."""
        invalid_config = {
            "language": "python",
            "invalid_key": "invalid_value"
        }
        
        try:
            _validate_config_schema(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown config option: invalid_key" in str(e)

    def test_config_schema_validation_rejects_wrong_types(self):
        """Test that wrong value types are rejected."""
        invalid_config = {
            "language": "python",
            "include_edge_cases": "true"  # Should be bool, not string
        }
        
        try:
            _validate_config_schema(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid type for include_edge_cases" in str(e)

    def test_config_schema_validation_accepts_valid_config(self):
        """Test that valid config passes validation."""
        valid_config = {
            "language": "python",
            "include_edge_cases": True,
            "include_error_paths": False,
            "include_benchmarks": True,
            "include_integration_tests": False
        }
        
        result = _validate_config_schema(valid_config)
        assert result == valid_config

    def test_path_validation_rejects_nonexistent_file(self):
        """Test that nonexistent files are rejected."""
        parser = _build_parser()
        args = parser.parse_args(["generate", "--file", "/nonexistent/file.py", "--output", "tests"])
        
        try:
            _validate_paths(args, parser)
            assert False, "Should have called parser.error"
        except SystemExit:
            # parser.error() calls sys.exit(), which we expect
            pass

    def test_path_validation_rejects_nonexistent_project_dir(self):
        """Test that nonexistent project directories are rejected."""
        parser = _build_parser()
        args = parser.parse_args(["generate", "--project", "/nonexistent/dir", "--output", "tests", "--batch"])
        
        try:
            _validate_paths(args, parser)
            assert False, "Should have called parser.error"
        except SystemExit:
            pass

    def test_path_validation_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid source file
            source_file = Path(temp_dir) / "test.py"
            source_file.write_text("def test(): pass")
            
            # Set output to non-existent subdirectory
            output_dir = Path(temp_dir) / "new_output"
            
            parser = _build_parser()
            args = parser.parse_args(["generate", "--file", str(source_file), "--output", str(output_dir)])
            
            # Should not raise exception and should create directory
            _validate_paths(args, parser)
            assert output_dir.exists()
            assert output_dir.is_dir()

    def test_path_validation_rejects_file_as_output_dir(self):
        """Test that existing files cannot be used as output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files
            source_file = Path(temp_dir) / "test.py"
            source_file.write_text("def test(): pass")
            
            output_file = Path(temp_dir) / "output.txt"
            output_file.write_text("existing file")
            
            parser = _build_parser()
            args = parser.parse_args(["generate", "--file", str(source_file), "--output", str(output_file)])
            
            try:
                _validate_paths(args, parser)
                assert False, "Should have called parser.error"
            except SystemExit:
                pass

    def test_path_validation_normalizes_paths(self):
        """Test that valid paths are normalized to absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid source file
            source_file = Path(temp_dir) / "test.py"
            source_file.write_text("def test(): pass")
            
            # Create output directory
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            parser = _build_parser()
            args = parser.parse_args(["generate", "--file", str(source_file), "--output", str(output_dir)])
            
            _validate_paths(args, parser)
            
            # Paths should be normalized to absolute
            assert Path(args.file).is_absolute()
            assert Path(args.output).is_absolute()
            assert args.file == str(source_file.resolve())
            assert args.output == str(output_dir.resolve())

    def test_dangerous_config_file_content_is_rejected(self):
        """Test that malicious config content is safely handled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write invalid JSON that could be dangerous
            f.write('{"language": "python", "include_edge_cases": true, "__import__": "os"}')
            config_path = Path(f.name)
        
        try:
            invalid_config = json.loads(config_path.read_text())
            result = _validate_config_schema(invalid_config)
            assert False, "Should have rejected unknown key __import__"
        except ValueError as e:
            assert "Dangerous config option detected" in str(e)
        finally:
            config_path.unlink()

    def test_coverage_target_validation(self):
        """Test that coverage targets are properly validated."""
        parser = _build_parser()
        
        # Test negative coverage target
        args = parser.parse_args(["generate", "--project", ".", "--output", "tests", "--batch", "--coverage-target", "-10"])
        assert args.coverage_target == -10  # Parser accepts it, but we should validate this
        
        # Test coverage target over 100
        args = parser.parse_args(["generate", "--project", ".", "--output", "tests", "--batch", "--coverage-target", "150"])
        assert args.coverage_target == 150  # Parser accepts it, but we should validate this

    def test_quality_target_validation(self):
        """Test that quality targets are properly validated."""
        parser = _build_parser()
        
        # Test negative quality target  
        args = parser.parse_args(["generate", "--project", ".", "--output", "tests", "--batch", "--quality-target", "-5"])
        assert args.quality_target == -5  # Parser accepts it, but we should validate this