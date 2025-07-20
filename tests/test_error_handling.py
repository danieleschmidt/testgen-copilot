"""Test comprehensive error handling across all modules."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from testgen_copilot.generator import TestGenerator, GenerationConfig
from testgen_copilot.security import SecurityScanner
from testgen_copilot.coverage import CoverageAnalyzer
from testgen_copilot.quality import TestQualityScorer


class TestGeneratorErrorHandling:
    """Test error handling in TestGenerator."""

    def test_generate_tests_nonexistent_file(self, tmp_path):
        """Test handling of non-existent source file."""
        generator = TestGenerator()
        nonexistent = tmp_path / "nonexistent.py"
        
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            generator.generate_tests(nonexistent, tmp_path)

    def test_generate_tests_directory_as_file(self, tmp_path):
        """Test handling when source path is a directory."""
        generator = TestGenerator()
        
        with pytest.raises(ValueError, match="Path is not a file"):
            generator.generate_tests(tmp_path, tmp_path)

    def test_generate_tests_unreadable_file(self, tmp_path):
        """Test handling of unreadable source file."""
        generator = TestGenerator()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        # Mock file read to raise PermissionError
        with patch.object(Path, 'read_text', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError, match="Cannot read source file"):
                generator.generate_tests(source_file, tmp_path)

    def test_generate_tests_invalid_encoding(self, tmp_path):
        """Test handling of files with invalid encoding."""
        generator = TestGenerator()
        source_file = tmp_path / "source.py"
        
        # Create file with invalid UTF-8
        with open(source_file, 'wb') as f:
            f.write(b'\xff\xfe\x00\x00invalid')
        
        with pytest.raises(ValueError, match="invalid text encoding"):
            generator.generate_tests(source_file, tmp_path)

    def test_generate_tests_syntax_error(self, tmp_path):
        """Test handling of source files with syntax errors."""
        generator = TestGenerator()
        source_file = tmp_path / "source.py"
        source_file.write_text("def invalid syntax here")
        
        with pytest.raises(SyntaxError, match="Cannot parse.*syntax error"):
            generator.generate_tests(source_file, tmp_path)

    def test_generate_tests_readonly_output_dir(self, tmp_path):
        """Test handling when output directory cannot be created."""
        generator = TestGenerator()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        # Mock mkdir to raise PermissionError
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Cannot create output directory"):
                generator.generate_tests(source_file, tmp_path / "output")

    def test_generate_tests_readonly_output_file(self, tmp_path):
        """Test handling when output file cannot be written."""
        generator = TestGenerator()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        # Mock file write to raise PermissionError
        with patch.object(Path, 'write_text', side_effect=PermissionError("Write denied")):
            with pytest.raises(PermissionError, match="Cannot write test file"):
                generator.generate_tests(source_file, tmp_path)

    def test_unsupported_language(self, tmp_path):
        """Test handling of unsupported programming language."""
        config = GenerationConfig(language="unsupported")
        generator = TestGenerator(config)
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        with pytest.raises(ValueError, match="Unsupported language: unsupported"):
            generator.generate_tests(source_file, tmp_path)


class TestSecurityScannerErrorHandling:
    """Test error handling in SecurityScanner."""

    def test_scan_nonexistent_file(self, tmp_path):
        """Test scanning non-existent file."""
        scanner = SecurityScanner()
        nonexistent = tmp_path / "nonexistent.py"
        
        report = scanner.scan_file(nonexistent)
        assert len(report.issues) == 1
        assert "File not found" in report.issues[0].message

    def test_scan_directory_as_file(self, tmp_path):
        """Test scanning a directory instead of file."""
        scanner = SecurityScanner()
        
        report = scanner.scan_file(tmp_path)
        assert len(report.issues) == 1
        assert "Path is not a file" in report.issues[0].message

    def test_scan_unreadable_file(self, tmp_path):
        """Test scanning unreadable file."""
        scanner = SecurityScanner()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        with patch.object(Path, 'read_text', side_effect=PermissionError("Access denied")):
            report = scanner.scan_file(source_file)
            assert len(report.issues) == 1
            assert "Cannot read file" in report.issues[0].message

    def test_scan_invalid_encoding(self, tmp_path):
        """Test scanning file with invalid encoding."""
        scanner = SecurityScanner()
        source_file = tmp_path / "source.py"
        
        # Create file with invalid UTF-8
        with open(source_file, 'wb') as f:
            f.write(b'\xff\xfe\x00\x00invalid')
        
        report = scanner.scan_file(source_file)
        assert len(report.issues) == 1
        assert "File encoding error" in report.issues[0].message

    def test_scan_syntax_error(self, tmp_path):
        """Test scanning file with syntax error."""
        scanner = SecurityScanner()
        source_file = tmp_path / "source.py"
        source_file.write_text("def invalid syntax")
        
        report = scanner.scan_file(source_file)
        assert len(report.issues) == 1
        assert "Syntax error" in report.issues[0].message

    def test_scan_project_nonexistent(self, tmp_path):
        """Test scanning non-existent project directory."""
        scanner = SecurityScanner()
        nonexistent = tmp_path / "nonexistent"
        
        reports = scanner.scan_project(nonexistent)
        assert len(reports) == 1
        assert "Project path not found" in reports[0].issues[0].message

    def test_scan_project_not_directory(self, tmp_path):
        """Test scanning file instead of project directory."""
        scanner = SecurityScanner()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        
        reports = scanner.scan_project(source_file)
        assert len(reports) == 1
        assert "Path is not a directory" in reports[0].issues[0].message

    def test_scan_project_no_python_files(self, tmp_path):
        """Test scanning project with no Python files."""
        scanner = SecurityScanner()
        
        reports = scanner.scan_project(tmp_path)
        assert len(reports) == 1
        assert "No Python files found" in reports[0].issues[0].message


class TestCoverageAnalyzerErrorHandling:
    """Test error handling in CoverageAnalyzer."""

    def test_analyze_nonexistent_source(self, tmp_path):
        """Test analyzing non-existent source file."""
        analyzer = CoverageAnalyzer()
        nonexistent = tmp_path / "nonexistent.py"
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        coverage = analyzer.analyze(nonexistent, tests_dir)
        assert coverage == 0.0

    def test_analyze_source_not_file(self, tmp_path):
        """Test analyzing directory instead of source file."""
        analyzer = CoverageAnalyzer()
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        coverage = analyzer.analyze(tmp_path, tests_dir)
        assert coverage == 0.0

    def test_analyze_nonexistent_tests_dir(self, tmp_path):
        """Test analyzing with non-existent tests directory."""
        analyzer = CoverageAnalyzer()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        nonexistent_tests = tmp_path / "nonexistent"
        
        coverage = analyzer.analyze(source_file, nonexistent_tests)
        assert coverage == 0.0

    def test_analyze_tests_not_directory(self, tmp_path):
        """Test analyzing with file instead of tests directory."""
        analyzer = CoverageAnalyzer()
        source_file = tmp_path / "source.py"
        source_file.write_text("def test(): pass")
        not_dir = tmp_path / "notdir.py"
        not_dir.write_text("not a directory")
        
        coverage = analyzer.analyze(source_file, not_dir)
        assert coverage == 0.0

    def test_analyze_syntax_error_source(self, tmp_path):
        """Test analyzing source file with syntax error."""
        analyzer = CoverageAnalyzer()
        source_file = tmp_path / "source.py"
        source_file.write_text("def invalid syntax")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        coverage = analyzer.analyze(source_file, tests_dir)
        assert coverage == 0.0

    def test_uncovered_functions_invalid_paths(self, tmp_path):
        """Test uncovered_functions with invalid paths."""
        analyzer = CoverageAnalyzer()
        nonexistent = tmp_path / "nonexistent.py"
        
        uncovered = analyzer.uncovered_functions(nonexistent, tmp_path)
        assert uncovered == set()


class TestQualityScorerErrorHandling:
    """Test error handling in TestQualityScorer."""

    def test_score_nonexistent_tests_dir(self, tmp_path):
        """Test scoring non-existent tests directory."""
        scorer = TestQualityScorer()
        nonexistent = tmp_path / "nonexistent"
        
        score = scorer.score(nonexistent)
        assert score == 0.0

    def test_score_file_instead_of_directory(self, tmp_path):
        """Test scoring file instead of directory."""
        scorer = TestQualityScorer()
        not_dir = tmp_path / "notdir.py"
        not_dir.write_text("not a directory")
        
        score = scorer.score(not_dir)
        assert score == 0.0

    def test_score_no_test_files(self, tmp_path):
        """Test scoring directory with no test files."""
        scorer = TestQualityScorer()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        score = scorer.score(empty_dir)
        assert score == 100.0  # No tests to score

    def test_score_syntax_error_in_tests(self, tmp_path):
        """Test scoring with syntax error in test file."""
        scorer = TestQualityScorer()
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        # Create test file with syntax error
        test_file = tests_dir / "test_invalid.py"
        test_file.write_text("def test_invalid syntax error")
        
        # Should skip the invalid file and return 100% (no valid tests found)
        score = scorer.score(tests_dir)
        assert score == 100.0

    def test_low_quality_tests_invalid_directory(self, tmp_path):
        """Test low_quality_tests with invalid directory."""
        scorer = TestQualityScorer()
        nonexistent = tmp_path / "nonexistent"
        
        lacking = scorer.low_quality_tests(nonexistent)
        assert lacking == set()


class TestIntegrationErrorHandling:
    """Test error handling in integrated scenarios."""

    def test_generator_with_all_error_conditions(self, tmp_path):
        """Test generator handles multiple error conditions gracefully."""
        generator = TestGenerator()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            generator.generate_tests(tmp_path / "missing.py", tmp_path)
        
        # Test with invalid syntax
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("def broken syntax")
        
        with pytest.raises(SyntaxError):
            generator.generate_tests(invalid_file, tmp_path)

    def test_scanner_handles_mixed_file_conditions(self, tmp_path):
        """Test scanner handles mix of valid and invalid files."""
        scanner = SecurityScanner()
        
        # Create mix of files
        valid_file = tmp_path / "valid.py"
        valid_file.write_text("def safe_function(): pass")
        
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("def broken syntax")
        
        reports = scanner.scan_project(tmp_path)
        
        # Should get reports for both files, with error for invalid one
        assert len(reports) == 2
        valid_report = next(r for r in reports if r.path.name == "valid.py")
        invalid_report = next(r for r in reports if r.path.name == "invalid.py")
        
        assert len(valid_report.issues) == 0  # Valid file, no issues
        assert len(invalid_report.issues) == 1  # Invalid file, syntax error
        assert "Syntax error" in invalid_report.issues[0].message