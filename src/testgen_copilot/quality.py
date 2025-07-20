"""Utilities to score the quality of test files."""

from __future__ import annotations

import ast
import logging
from pathlib import Path


class TestQualityScorer:
    """Estimate quality of tests based on presence of assertions."""

    def score(self, tests_dir: str | Path) -> float:
        """Return percentage of test functions containing ``assert`` statements."""
        logger = logging.getLogger(__name__)
        
        try:
            tests = Path(tests_dir)
            
            # Validate tests directory
            if not tests.exists():
                logger.warning(f"Tests directory not found: {tests}")
                return 0.0
            
            if not tests.is_dir():
                logger.error(f"Tests path is not a directory: {tests}")
                return 0.0
            
            total = 0
            with_assert = 0
            test_files = list(tests.rglob("test_*.py"))
            
            if not test_files:
                logger.warning(f"No test files found in {tests}")
                return 100.0  # No tests to score, consider perfect
            
            logger.debug(f"Analyzing quality of {len(test_files)} test files")
            
            for path in test_files:
                try:
                    try:
                        content = path.read_text()
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Cannot read test file {path}: {e}")
                        continue
                    except UnicodeDecodeError as e:
                        logger.warning(f"Encoding error in test file {path}: {e}")
                        continue
                    
                    try:
                        tree = ast.parse(content)
                    except SyntaxError as e:
                        logger.warning(f"Syntax error in test file {path} at line {e.lineno}: {e.msg}")
                        continue
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                            total += 1
                            if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                                with_assert += 1
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze test file {path}: {e}")
                    continue
            
            if total == 0:
                logger.info("No test functions found")
                return 100.0
            
            score = (with_assert / total) * 100
            logger.debug(f"Test quality score: {score:.1f}% ({with_assert}/{total} functions with assertions)")
            return score
            
        except Exception as e:
            logger.error(f"Failed to score test quality for {tests_dir}: {e}")
            return 0.0

    def low_quality_tests(self, tests_dir: str | Path) -> set[str]:
        """Return names of test functions lacking assertions."""
        logger = logging.getLogger(__name__)
        
        try:
            tests = Path(tests_dir)
            
            # Validate tests directory
            if not tests.exists() or not tests.is_dir():
                logger.error(f"Invalid tests directory: {tests}")
                return set()
            
            lacking: set[str] = set()
            test_files = list(tests.rglob("test_*.py"))
            
            if not test_files:
                logger.warning(f"No test files found in {tests}")
                return set()
            
            logger.debug(f"Finding low-quality tests in {len(test_files)} files")
            
            for path in test_files:
                try:
                    try:
                        content = path.read_text()
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Cannot read test file {path}: {e}")
                        continue
                    except UnicodeDecodeError as e:
                        logger.warning(f"Encoding error in test file {path}: {e}")
                        continue
                    
                    try:
                        tree = ast.parse(content)
                    except SyntaxError as e:
                        logger.warning(f"Syntax error in test file {path} at line {e.lineno}: {e.msg}")
                        continue
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                            if not any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                                lacking.add(node.name)
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze test file {path}: {e}")
                    continue
            
            logger.debug(f"Found {len(lacking)} low-quality test functions")
            return lacking
            
        except Exception as e:
            logger.error(f"Failed to find low-quality tests in {tests_dir}: {e}")
            return set()
