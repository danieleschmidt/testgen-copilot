"""Simple code coverage estimation utilities."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Iterable


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
            content = path.read_text()
        except (OSError, PermissionError) as e:
            logger.error(f"Cannot read file {path}: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"File encoding error in {path}: {e}")
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
