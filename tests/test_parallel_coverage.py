"""Tests for parallel coverage analysis functionality."""

import multiprocessing
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from testgen_copilot.coverage import (
    ParallelCoverageAnalyzer, 
    CoverageResult, 
    _analyze_single_file,
    CoverageAnalyzer
)


class TestParallelCoverageAnalyzer:
    """Test suite for ParallelCoverageAnalyzer."""

    def test_initialization_default_workers(self):
        """Test that analyzer initializes with appropriate default worker count."""
        analyzer = ParallelCoverageAnalyzer()
        expected_workers = min(multiprocessing.cpu_count() or 1, 8)
        assert analyzer.max_workers == expected_workers

    def test_initialization_custom_workers(self):
        """Test that analyzer respects custom worker count."""
        analyzer = ParallelCoverageAnalyzer(max_workers=4)
        assert analyzer.max_workers == 4

    def test_analyze_project_parallel_basic(self, tmp_path):
        """Test basic parallel project analysis functionality."""
        # Create test project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        # Create source files with different coverage levels
        (project_dir / "fully_covered.py").write_text(
            """
def covered_function():
    return "covered"
"""
        )
        
        (project_dir / "partially_covered.py").write_text(
            """
def covered_function():
    return "covered"

def uncovered_function():
    return "uncovered"
"""
        )
        
        (project_dir / "uncovered.py").write_text(
            """
def uncovered_function():
    return "uncovered"
"""
        )

        # Create test files
        (tests_dir / "test_fully_covered.py").write_text(
            """
from fully_covered import covered_function

def test_covered_function():
    assert covered_function() == "covered"
"""
        )
        
        (tests_dir / "test_partially_covered.py").write_text(
            """
from partially_covered import covered_function

def test_covered_function():
    assert covered_function() == "covered"
"""
        )

        # Run parallel analysis
        analyzer = ParallelCoverageAnalyzer(max_workers=2)
        results = analyzer.analyze_project_parallel(
            project_dir=project_dir,
            tests_dir=tests_dir,
            target_coverage=80.0
        )

        # Verify results
        assert len(results) == 2  # partially_covered.py and uncovered.py should be below 80%
        
        file_results = {result.file_path: result for result in results}
        assert str(project_dir / "partially_covered.py") in file_results
        assert str(project_dir / "uncovered.py") in file_results
        
        # Check specific coverage values
        partially_covered_result = file_results[str(project_dir / "partially_covered.py")]
        assert partially_covered_result.coverage_percentage == 50.0
        assert "uncovered_function" in partially_covered_result.uncovered_functions
        
        uncovered_result = file_results[str(project_dir / "uncovered.py")]
        assert uncovered_result.coverage_percentage == 0.0
        assert "uncovered_function" in uncovered_result.uncovered_functions

    def test_analyze_project_parallel_with_progress(self, tmp_path):
        """Test parallel analysis with progress callback."""
        # Create simple project
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        # Create multiple source files
        for i in range(5):
            (project_dir / f"module_{i}.py").write_text(
                f"""
def function_{i}():
    return {i}
"""
            )

        # Track progress calls
        progress_calls = []
        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        analyzer = ParallelCoverageAnalyzer(max_workers=2)
        results = analyzer.analyze_project_parallel(
            project_dir=project_dir,
            tests_dir=tests_dir,
            target_coverage=50.0,
            progress_callback=progress_callback
        )

        # Verify progress was tracked
        assert len(progress_calls) == 5
        assert progress_calls[-1] == (5, 5)  # Final call should be (completed=5, total=5)

    def test_analyze_project_parallel_no_files(self, tmp_path):
        """Test parallel analysis with no Python files."""
        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        analyzer = ParallelCoverageAnalyzer()
        results = analyzer.analyze_project_parallel(
            project_dir=project_dir,
            tests_dir=tests_dir,
            target_coverage=80.0
        )

        assert results == []

    def test_analyze_project_parallel_excludes_test_files(self, tmp_path):
        """Test that parallel analysis excludes test files from analysis."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        # Create source file
        (project_dir / "source.py").write_text(
            """
def source_function():
    return "source"
"""
        )

        # Create test file that should be excluded
        (tests_dir / "test_source.py").write_text(
            """
def test_function():
    return "test"
"""
        )

        analyzer = ParallelCoverageAnalyzer()
        results = analyzer.analyze_project_parallel(
            project_dir=project_dir,
            tests_dir=tests_dir,
            target_coverage=50.0
        )

        # Should only find the source file, not the test file
        assert len(results) == 1
        assert "source.py" in results[0].file_path
        assert "test_source.py" not in results[0].file_path

    @patch('testgen_copilot.coverage.multiprocessing.Pool')
    def test_multiprocessing_failure_fallback(self, mock_pool, tmp_path):
        """Test fallback to sequential processing when multiprocessing fails."""
        # Mock multiprocessing failure
        mock_pool.side_effect = Exception("Multiprocessing failed")

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        (project_dir / "source.py").write_text(
            """
def uncovered_function():
    return "uncovered"
"""
        )

        analyzer = ParallelCoverageAnalyzer()
        results = analyzer.analyze_project_parallel(
            project_dir=project_dir,
            tests_dir=tests_dir,
            target_coverage=50.0
        )

        # Should still get results via fallback
        assert len(results) == 1
        assert results[0].coverage_percentage == 0.0


class TestAnalyzeSingleFileWorker:
    """Test suite for the _analyze_single_file worker function."""

    def test_analyze_single_file_below_target(self, tmp_path):
        """Test worker function with file below coverage target."""
        source_file = tmp_path / "source.py"
        source_file.write_text(
            """
def covered_function():
    return "covered"

def uncovered_function():
    return "uncovered"
"""
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_source.py").write_text(
            """
from source import covered_function

def test_covered():
    assert covered_function() == "covered"
"""
        )

        result = _analyze_single_file((source_file, tests_dir, 80.0))

        assert result is not None
        assert isinstance(result, CoverageResult)
        assert result.coverage_percentage == 50.0
        assert "uncovered_function" in result.uncovered_functions
        assert result.total_functions == 2
        assert result.covered_functions == 1

    def test_analyze_single_file_above_target(self, tmp_path):
        """Test worker function with file above coverage target."""
        source_file = tmp_path / "source.py"
        source_file.write_text(
            """
def covered_function():
    return "covered"
"""
        )

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_source.py").write_text(
            """
from source import covered_function

def test_covered():
    assert covered_function() == "covered"
"""
        )

        result = _analyze_single_file((source_file, tests_dir, 80.0))

        # Should return None for files above target
        assert result is None

    def test_analyze_single_file_error_handling(self, tmp_path):
        """Test worker function error handling."""
        nonexistent_file = tmp_path / "nonexistent.py"
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        result = _analyze_single_file((nonexistent_file, tests_dir, 80.0))

        # Should return None on error
        assert result is None


class TestCoverageResult:
    """Test suite for CoverageResult dataclass."""

    def test_coverage_result_creation(self):
        """Test CoverageResult object creation."""
        result = CoverageResult(
            file_path="/path/to/file.py",
            coverage_percentage=75.0,
            uncovered_functions={"func1", "func2"},
            total_functions=4,
            covered_functions=2
        )

        assert result.file_path == "/path/to/file.py"
        assert result.coverage_percentage == 75.0
        assert result.uncovered_functions == {"func1", "func2"}
        assert result.total_functions == 4
        assert result.covered_functions == 2


class TestIntegrationWithExistingTests:
    """Integration tests to ensure parallel analyzer works with existing test suite."""

    def test_parallel_vs_sequential_consistency(self, tmp_path):
        """Test that parallel analyzer produces same results as sequential."""
        # Create test project
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()

        # Create source files
        (project_dir / "module1.py").write_text(
            """
def func1():
    return 1

def func2():
    return 2
"""
        )

        (project_dir / "module2.py").write_text(
            """
def func3():
    return 3
"""
        )

        # Create test file that covers only some functions
        (tests_dir / "test_module1.py").write_text(
            """
from module1 import func1

def test_func1():
    assert func1() == 1
"""
        )

        # Test with sequential analyzer (via fallback)
        with patch('testgen_copilot.coverage.multiprocessing.Pool') as mock_pool:
            mock_pool.side_effect = Exception("Force fallback")
            parallel_analyzer = ParallelCoverageAnalyzer()
            parallel_results = parallel_analyzer.analyze_project_parallel(
                project_dir=project_dir,
                tests_dir=tests_dir,
                target_coverage=80.0
            )

        # Test with regular CoverageAnalyzer
        sequential_analyzer = CoverageAnalyzer()
        sequential_failures = []
        for path in project_dir.rglob("*.py"):
            if tests_dir in path.parents:
                continue
            cov = sequential_analyzer.analyze(path, tests_dir)
            if cov < 80.0:
                missing = sequential_analyzer.uncovered_functions(path, tests_dir)
                sequential_failures.append((str(path.relative_to(project_dir)), cov, missing))

        # Results should be consistent
        assert len(parallel_results) == len(sequential_failures)
        
        # Convert parallel results to same format for comparison
        parallel_failures = [
            (str(Path(r.file_path).relative_to(project_dir)), r.coverage_percentage, r.uncovered_functions)
            for r in parallel_results
        ]
        
        # Sort both lists for comparison
        parallel_failures.sort(key=lambda x: x[0])
        sequential_failures.sort(key=lambda x: x[0])
        
        assert parallel_failures == sequential_failures