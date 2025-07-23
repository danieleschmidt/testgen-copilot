"""Simple code coverage estimation utilities."""

from __future__ import annotations

import ast
import logging
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .file_utils import safe_read_file, FileSizeError


@dataclass
class CoverageResult:
    """Result of coverage analysis for a single file."""
    file_path: str
    coverage_percentage: float
    uncovered_functions: set[str]
    total_functions: int
    covered_functions: int


def _analyze_single_file(args: Tuple[Path, Path, float]) -> Optional[CoverageResult]:
    """Worker function for multiprocessing coverage analysis.
    
    Args:
        args: Tuple of (source_file_path, tests_dir_path, target_coverage)
        
    Returns:
        CoverageResult if the file is below target coverage, None otherwise
    """
    source_path, tests_dir, target = args
    logger = logging.getLogger(__name__)
    
    try:
        analyzer = CoverageAnalyzer()
        coverage_pct = analyzer.analyze(source_path, tests_dir)
        
        if coverage_pct < target:
            uncovered = analyzer.uncovered_functions(source_path, tests_dir)
            
            # Calculate total functions
            try:
                all_functions = set(analyzer._functions_in_file(source_path))
                covered_count = len(all_functions) - len(uncovered)
                
                return CoverageResult(
                    file_path=str(source_path),
                    coverage_percentage=coverage_pct,
                    uncovered_functions=uncovered,
                    total_functions=len(all_functions),
                    covered_functions=covered_count
                )
            except Exception as e:
                logger.error(f"Failed to calculate function counts for {source_path}: {e}")
                return CoverageResult(
                    file_path=str(source_path),
                    coverage_percentage=coverage_pct,
                    uncovered_functions=uncovered,
                    total_functions=0,
                    covered_functions=0
                )
        return None
        
    except Exception as e:
        logger.error(f"Failed to analyze coverage for {source_path}: {e}")
        return None


class ParallelCoverageAnalyzer:
    """Parallel coverage analyzer using multiprocessing for improved performance."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel coverage analyzer.
        
        Args:
            max_workers: Maximum number of worker processes. Defaults to CPU count.
        """
        self.max_workers = max_workers or min(os.cpu_count() or 1, 8)  # Cap at 8 for memory efficiency
        self.logger = logging.getLogger(__name__)
    
    def analyze_project_parallel(
        self, 
        project_dir: Path, 
        tests_dir: Path, 
        target_coverage: float,
        progress_callback: Optional[callable] = None
    ) -> List[CoverageResult]:
        """Analyze coverage for all Python files in project using multiprocessing.
        
        Args:
            project_dir: Root directory of the project
            tests_dir: Directory containing test files
            target_coverage: Coverage threshold (0-100)
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            List of CoverageResult for files below target coverage
        """
        self.logger.info(f"Starting parallel coverage analysis with {self.max_workers} workers")
        
        # Find all Python files, excluding test files
        python_files = [
            path for path in project_dir.rglob("*.py")
            if tests_dir not in path.parents and path.name != "__pycache__"
        ]
        
        if not python_files:
            self.logger.warning(f"No Python files found in {project_dir}")
            return []
        
        self.logger.info(f"Analyzing {len(python_files)} Python files")
        
        # Prepare tasks for multiprocessing
        tasks = [(file_path, tests_dir, target_coverage) for file_path in python_files]
        
        try:
            # Use multiprocessing pool for parallel analysis
            with multiprocessing.Pool(self.max_workers) as pool:
                if progress_callback:
                    # Use imap for progress reporting
                    results = []
                    for i, result in enumerate(pool.imap(_analyze_single_file, tasks)):
                        if result is not None:
                            results.append(result)
                        progress_callback(i + 1, len(tasks))
                else:
                    # Use map for batch processing
                    all_results = pool.map(_analyze_single_file, tasks)
                    results = [r for r in all_results if r is not None]
            
            self.logger.info(f"Coverage analysis complete. Found {len(results)} files below target")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel coverage analysis failed: {e}")
            # Fallback to sequential processing
            return self._analyze_sequential_fallback(tasks)
    
    def _analyze_sequential_fallback(self, tasks: List[Tuple[Path, Path, float]]) -> List[CoverageResult]:
        """Fallback to sequential processing if multiprocessing fails."""
        self.logger.warning("Falling back to sequential coverage analysis")
        results = []
        
        for task in tasks:
            try:
                result = _analyze_single_file(task)
                if result is not None:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {task[0]}: {e}")
                continue
        
        return results


class CoverageAnalyzer:
    """Estimate coverage of tests over a source file."""

    def analyze(self, source_path: str | Path, tests_dir: str | Path) -> float:
        """Return the percentage of source functions referenced in tests."""
        logger = logging.getLogger(__name__)
        
        try:
            src = Path(source_path)
            tests = Path(tests_dir)
            
            # Validate input paths
            if not src.exists():
                logger.error(f"Source file not found: {src}")
                return 0.0
            
            if not src.is_file():
                logger.error(f"Source path is not a file: {src}")
                return 0.0
            
            if not tests.exists():
                logger.warning(f"Tests directory not found: {tests}")
                return 0.0
            
            if not tests.is_dir():
                logger.error(f"Tests path is not a directory: {tests}")
                return 0.0
            
            try:
                func_names = self._functions_in_file(src)
            except Exception as e:
                logger.error(f"Failed to parse functions from {src}: {e}")
                return 0.0
            
            if not func_names:
                logger.debug(f"No functions found in {src}")
                return 100.0
            
            try:
                covered = self._functions_used_in_tests(tests, func_names)
            except Exception as e:
                logger.error(f"Failed to analyze test coverage for {src}: {e}")
                return 0.0
            
            coverage = (len(covered) / len(func_names)) * 100
            logger.debug(f"Coverage for {src}: {coverage:.1f}% ({len(covered)}/{len(func_names)} functions)")
            return coverage
            
        except Exception as e:
            logger.error(f"Coverage analysis failed for {source_path}: {e}")
            return 0.0

    def uncovered_functions(
        self, source_path: str | Path, tests_dir: str | Path
    ) -> set[str]:
        """Return names of functions in ``source_path`` not referenced by tests."""
        logger = logging.getLogger(__name__)
        
        try:
            src = Path(source_path)
            tests = Path(tests_dir)
            
            # Validate input paths
            if not src.exists() or not src.is_file():
                logger.error(f"Invalid source file: {src}")
                return set()
            
            if not tests.exists() or not tests.is_dir():
                logger.warning(f"Invalid tests directory: {tests}")
                return set()
            
            try:
                func_names = self._functions_in_file(src)
            except Exception as e:
                logger.error(f"Failed to parse functions from {src}: {e}")
                return set()
            
            if not func_names:
                return set()
            
            try:
                covered = self._functions_used_in_tests(tests, func_names)
            except Exception as e:
                logger.error(f"Failed to analyze test usage for {src}: {e}")
                return set(func_names)  # Assume all uncovered on error
            
            uncovered = set(func_names) - covered
            logger.debug(f"Uncovered functions in {src}: {len(uncovered)} out of {len(func_names)}")
            return uncovered
            
        except Exception as e:
            logger.error(f"Failed to find uncovered functions for {source_path}: {e}")
            return set()

    @staticmethod
    def _functions_in_file(path: Path) -> list[str]:
        """Return all function and method names defined in ``path``."""
        logger = logging.getLogger(__name__)
        
        try:
            content = safe_read_file(path)
        except (FileNotFoundError, PermissionError, ValueError, FileSizeError, OSError) as e:
            # Errors are already logged by safe_read_file with structured context
            raise
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {path} at line {e.lineno}: {e.msg}")
            raise
        
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        
        logger.debug(f"Found {len(functions)} functions in {path}")
        return functions

    @staticmethod
    def _functions_used_in_tests(tests_dir: Path, func_names: Iterable[str]) -> set[str]:
        """Return the subset of ``func_names`` referenced within ``tests_dir``."""
        logger = logging.getLogger(__name__)
        
        names = set(func_names)
        covered: set[str] = set()
        test_files = list(tests_dir.rglob("test_*.py"))
        
        if not test_files:
            logger.warning(f"No test files found in {tests_dir}")
            return covered
        
        logger.debug(f"Analyzing {len(test_files)} test files for function usage")
        
        for test_file in test_files:
            try:
                try:
                    content = test_file.read_text()
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot read test file {test_file}: {e}")
                    continue
                except UnicodeDecodeError as e:
                    logger.warning(f"Encoding error in test file {test_file}: {e}")
                    continue
                
                try:
                    tree = ast.parse(content)
                except SyntaxError as e:
                    logger.warning(f"Syntax error in test file {test_file} at line {e.lineno}: {e.msg}")
                    continue
                
                # Track aliases from ``from x import y as z`` so calls to ``z`` are
                # associated with ``y`` when checking coverage.
                alias_map = {}
                try:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            for alias in node.names:
                                if alias.asname and alias.name in names:
                                    alias_map[alias.asname] = alias.name

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            if isinstance(node.ctx, ast.Load):
                                target = alias_map.get(node.id, node.id)
                                if target in names:
                                    covered.add(target)
                        elif isinstance(node, ast.Attribute):
                            if isinstance(node.ctx, ast.Load) and node.attr in names:
                                covered.add(node.attr)
                except Exception as e:
                    logger.warning(f"Error analyzing AST in {test_file}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to process test file {test_file}: {e}")
                continue
        
        logger.debug(f"Found {len(covered)} covered functions out of {len(names)}")
        return covered
