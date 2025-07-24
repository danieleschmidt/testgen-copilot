"""Simple code coverage estimation utilities."""

from __future__ import annotations

import ast
import logging
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .file_utils import safe_read_file, FileSizeError, safe_parse_ast
from .logging_config import get_coverage_logger
from .resource_limits import ResourceMonitor, ResourceLimits
from .ast_utils import safe_parse_ast, ASTParsingError

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
    logger = get_coverage_logger()
    
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
        self.logger = get_coverage_logger()
    
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
        logger = get_coverage_logger()
        
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
        logger = get_coverage_logger()
        
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
        logger = get_coverage_logger()
        
        try:
            tree, content = safe_parse_ast(path)
            content = safe_read_file(path)
        except (FileNotFoundError, PermissionError, ValueError, FileSizeError, OSError) as e:
            # Errors are already logged by safe_read_file with structured context
            raise
        
        try:
            tree = safe_parse_ast(content, path)
        except ASTParsingError as e:
            logger.error("Cannot parse source file", {
                "file_path": str(path),
                "error_message": str(e)
            })
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
        logger = get_coverage_logger()
        
        names = set(func_names)
        covered: set[str] = set()
        test_files = list(tests_dir.rglob("test_*.py"))
        
        if not test_files:
            logger.warning(f"No test files found in {tests_dir}")
            return covered
        
        logger.debug(f"Analyzing {len(test_files)} test files for function usage")
        
        for test_file in test_files:
            try:
                result = safe_parse_ast(test_file, raise_on_syntax_error=False)
                try:
                    content = test_file.read_text()
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot read test file {test_file}: {e}")
                    continue
                except UnicodeDecodeError as e:
                    logger.warning(f"Encoding error in test file {test_file}: {e}")
                    continue
                
                try:
                    tree = safe_parse_ast(content, test_file)
                except ASTParsingError as e:
                    logger.warning("Skipping test file due to parsing error", {
                        "file_path": str(test_file),
                        "error_message": str(e)
                    })
                    continue
                tree, content = result
            except (FileNotFoundError, PermissionError, ValueError, FileSizeError, OSError) as e:
                logger.warning(f"Cannot read test file {test_file}: {e}")
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
    
    def analyze_project_batch(self, project_path: Path, max_files: Optional[int] = None) -> List[CoverageResult]:
        """Analyze coverage for multiple files in a project with batch size limits."""
        logger = get_coverage_logger()
        resource_monitor = ResourceMonitor()
        
        # Find all Python files in the project
        python_files = list(project_path.rglob("*.py"))
        
        # Apply batch size limits
        if max_files is None:
            max_files = resource_monitor.limits.max_batch_size
        
        resource_monitor.validate_batch_size(len(python_files))
        
        if len(python_files) > max_files:
            logger.warning("Project has too many files, limiting batch", {
                "total_files": len(python_files),
                "max_files": max_files,
                "project_path": str(project_path)
            })
            python_files = python_files[:max_files]
        
        logger.info("Starting project batch analysis", {
            "project_path": str(project_path),
            "file_count": len(python_files),
            "max_batch_size": max_files
        })
        
        results = []
        for i, file_path in enumerate(python_files):
            # Check memory usage periodically
            if i % 50 == 0 and not resource_monitor.check_memory_usage():
                logger.error("Stopping batch analysis due to high memory usage", {
                    "processed_files": i,
                    "total_files": len(python_files)
                })
                raise MemoryError("Insufficient memory to continue batch analysis")
            
            try:
                # Analyze individual file coverage
                # For now, just create a placeholder result
                # In a real implementation, you'd analyze against test directories
                coverage_pct = self.analyze(file_path, project_path / "tests")
                
                if coverage_pct < 80:  # Default threshold
                    result = CoverageResult(
                        source_file=str(file_path),
                        coverage_percentage=coverage_pct,
                        uncovered_functions=self.uncovered_functions(file_path, project_path / "tests"),
                        total_functions=len(self._functions_in_file(file_path))
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning("Failed to analyze file in batch", {
                    "file_path": str(file_path),
                    "error_message": str(e)
                })
                continue
        
        logger.info("Project batch analysis completed", {
            "total_files_processed": len(python_files),
            "files_needing_coverage": len(results)
        })
        
        return results
