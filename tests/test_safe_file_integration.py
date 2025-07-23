"""Integration tests for safe file I/O across all modules."""

import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

from testgen_copilot.generator import TestGenerator, GenerationConfig
from testgen_copilot.coverage import CoverageAnalyzer
from testgen_copilot.quality import TestQualityScorer
from testgen_copilot.security import SecurityScanner
from testgen_copilot.file_utils import FileSizeError


class TestSafeFileIOIntegration:
    """Test that all modules properly use safe file I/O."""

    @pytest.fixture
    def sample_python_file(self, tmp_path):
        """Create a sample Python file for testing."""
        test_file = tmp_path / "sample.py"
        test_file.write_text("""
def add(a, b):
    '''Add two numbers.'''
    return a + b

def multiply(x, y):
    '''Multiply two numbers.'''  
    return x * y

def divide(num, denom):
    '''Divide two numbers.'''
    if denom == 0:
        raise ValueError("Cannot divide by zero")
    return num / denom
""")
        return test_file

    @pytest.fixture
    def sample_test_file(self, tmp_path):
        """Create a sample test file for testing."""
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("""
import pytest

def test_add():
    from sample import add
    assert add(2, 3) == 5
    assert add(0, 0) == 0

def test_multiply():
    from sample import multiply
    assert multiply(4, 5) == 20

@pytest.mark.parametrize("num,denom,expected", [
    (10, 2, 5),
    (15, 3, 5),
])
def test_divide_parameterized(num, denom, expected):
    from sample import divide
    assert divide(num, denom) == expected
""")
        return test_file

    def test_generator_handles_file_size_limit(self, tmp_path):
        """Test that TestGenerator respects file size limits."""
        large_file = tmp_path / "large.py"
        
        # Mock file size to be larger than default limit
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 11 * 1024 * 1024  # 11MB
            
            generator = TestGenerator(GenerationConfig(language="python"))
            
            with pytest.raises(FileSizeError) as exc_info:
                generator.generate_tests(large_file, tmp_path / "tests")
            
            assert "too large" in str(exc_info.value).lower()

    def test_generator_handles_nonexistent_file(self, tmp_path):
        """Test that TestGenerator handles nonexistent files gracefully."""
        nonexistent_file = tmp_path / "does_not_exist.py"
        
        generator = TestGenerator(GenerationConfig(language="python"))
        
        with pytest.raises(FileNotFoundError) as exc_info:
            generator.generate_tests(nonexistent_file, tmp_path / "tests")
        
        assert "not found" in str(exc_info.value).lower()

    def test_generator_handles_permission_error(self, sample_python_file, tmp_path):
        """Test that TestGenerator handles permission errors gracefully."""
        generator = TestGenerator(GenerationConfig(language="python"))
        
        with patch("testgen_copilot.file_utils.Path.read_text") as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError) as exc_info:
                generator.generate_tests(sample_python_file, tmp_path / "tests")
            
            assert "permission denied" in str(exc_info.value).lower()

    def test_coverage_analyzer_handles_file_errors(self, tmp_path):
        """Test that CoverageAnalyzer handles file errors gracefully."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        # Create a file that will cause read errors
        bad_file = src_dir / "bad.py"
        bad_file.write_text("def test(): pass")
        
        analyzer = CoverageAnalyzer()
        
        with patch("testgen_copilot.file_utils.safe_read_file") as mock_read:
            mock_read.side_effect = FileSizeError("File too large")
            
            # Should handle the error gracefully and not crash
            result = analyzer.analyze_coverage(str(src_dir), str(tests_dir))
            # Coverage should be 0% due to file read error
            assert result == 0.0

    def test_quality_scorer_handles_file_errors(self, tmp_path):
        """Test that TestQualityScorer handles file errors gracefully."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        # Create a test file that will cause read errors
        bad_file = tests_dir / "test_bad.py"
        bad_file.write_text("def test_something(): pass")
        
        scorer = TestQualityScorer()
        
        with patch("testgen_copilot.file_utils.safe_read_file") as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")
            
            # Should handle the error gracefully and return default score
            result = scorer.score(str(tests_dir))
            # Should return 100% since no files could be analyzed
            assert result == 100.0

    def test_security_scanner_handles_file_errors(self, tmp_path):
        """Test that SecurityScanner handles file errors gracefully."""
        test_file = tmp_path / "secure.py"
        test_file.write_text("def secure_function(): pass")
        
        scanner = SecurityScanner()
        
        with patch("testgen_copilot.file_utils.safe_read_file") as mock_read:
            mock_read.side_effect = FileSizeError("File too large (15.0MB, max: 10MB)")
            
            # Should return a security report with the error
            report = scanner.scan_file(str(test_file))
            
            assert report.file_path == test_file
            assert len(report.issues) == 1
            assert "too large" in report.issues[0].message.lower()

    def test_full_workflow_with_safe_file_io(self, sample_python_file, tmp_path):
        """Test full workflow using safe file I/O across all modules."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        # Generate tests using TestGenerator
        generator = TestGenerator(GenerationConfig(language="python"))
        test_file = generator.generate_tests(sample_python_file, tests_dir)
        
        # Verify test file was created
        assert test_file.exists()
        assert test_file.name == "test_sample.py"
        
        # Run security scan on original file
        scanner = SecurityScanner()
        security_report = scanner.scan_file(str(sample_python_file))
        
        # Should have scanned successfully
        assert security_report.file_path == sample_python_file
        # Should have minimal issues for this simple code
        assert len(security_report.issues) <= 2  # Might have input validation warnings
        
        # Analyze test quality
        scorer = TestQualityScorer()
        quality_score = scorer.score(str(tests_dir))
        
        # Should have reasonable quality score
        assert 0 <= quality_score <= 100
        
        # Analyze coverage
        analyzer = CoverageAnalyzer()
        coverage = analyzer.analyze_coverage(str(sample_python_file.parent), str(tests_dir))
        
        # Should have some coverage
        assert coverage >= 0

    def test_file_size_limits_respected_across_modules(self, tmp_path):
        """Test that file size limits are respected across all modules."""
        large_file = tmp_path / "large.py"
        large_file.write_text("def test(): pass")
        
        # Mock large file size
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 12 * 1024 * 1024  # 12MB
            
            # TestGenerator should reject large files
            generator = TestGenerator(GenerationConfig(language="python"))
            with pytest.raises(FileSizeError):
                generator.generate_tests(large_file, tmp_path / "tests")
            
            # SecurityScanner should handle large files gracefully
            scanner = SecurityScanner()
            report = scanner.scan_file(str(large_file))
            assert "too large" in report.issues[0].message.lower()
            
            # CoverageAnalyzer should handle large files gracefully
            analyzer = CoverageAnalyzer()
            # Should not crash, might return 0% coverage
            result = analyzer.analyze_coverage(str(large_file.parent), str(tmp_path / "tests"))
            assert isinstance(result, (int, float))
            
            # TestQualityScorer should handle large files gracefully
            scorer = TestQualityScorer()
            result = scorer.score(str(tmp_path))
            assert isinstance(result, (int, float))

    def test_unicode_handling_across_modules(self, tmp_path):
        """Test that Unicode decode errors are handled properly."""
        bad_file = tmp_path / "bad_encoding.py"
        # Write binary data that will cause UnicodeDecodeError
        bad_file.write_bytes(b'\xff\xfe\x80\x81')
        
        # TestGenerator should handle Unicode errors
        generator = TestGenerator(GenerationConfig(language="python"))
        with pytest.raises(ValueError) as exc_info:
            generator.generate_tests(bad_file, tmp_path / "tests")
        assert "encoding" in str(exc_info.value).lower()
        
        # SecurityScanner should handle Unicode errors gracefully
        scanner = SecurityScanner()
        report = scanner.scan_file(str(bad_file))
        assert "encoding" in report.issues[0].message.lower()
        
        # Other modules should handle gracefully in their loops
        # (they skip files with errors and continue)