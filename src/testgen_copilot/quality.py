"""Utilities to score the quality of test files."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from .file_utils import safe_read_file, FileSizeError, safe_parse_ast


class TestQualityScorer:
    """Estimate quality of tests based on presence of assertions and advanced patterns."""

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
            
            total_functions = 0
            functions_with_assert = 0
            test_files = list(tests.rglob("test_*.py"))
            
            if not test_files:
                logger.warning(f"No test files found in {tests}")
                return 100.0  # No tests to score, consider perfect
            
            logger.debug(f"Analyzing quality of {len(test_files)} test files")
            
            for path in test_files:
                try:
                    result = safe_parse_ast(path, raise_on_syntax_error=False)
                    if result is None:
                        # safe_parse_ast already logged the syntax error
                        continue
                    tree, content = result
                    
                    test_functions = self._find_test_functions(tree)
                    for func_info in test_functions:
                        total_functions += 1  # Count functions, not test cases
                        if func_info['has_assertions']:
                            functions_with_assert += 1
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze test file {path}: {e}")
                    continue
            
            if total_functions == 0:
                logger.info("No test functions found")
                return 100.0
            
            score = (functions_with_assert / total_functions) * 100
            logger.debug(f"Test quality score: {score:.1f}% ({functions_with_assert}/{total_functions} functions with assertions)")
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
                    result = safe_parse_ast(path, raise_on_syntax_error=False)
                    if result is None:
                        # safe_parse_ast already logged the syntax error
                        continue
                    tree, content = result
                    
                    test_functions = self._find_test_functions(tree)
                    for func_info in test_functions:
                        if not func_info['has_assertions']:
                            lacking.add(func_info['name'])
                                
                except Exception as e:
                    logger.warning(f"Failed to analyze test file {path}: {e}")
                    continue
            
            logger.debug(f"Found {len(lacking)} low-quality test functions")
            return lacking
            
        except Exception as e:
            logger.error(f"Failed to find low-quality tests in {tests_dir}: {e}")
            return set()
    
    @staticmethod
    def _find_test_functions(tree: ast.AST) -> list[dict]:
        """Find and analyze test functions, including parameterized tests."""
        logger = logging.getLogger(__name__)
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                # Check for assertions in the function
                has_assertions = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                
                # Check for pytest.mark.parametrize decorator
                param_count = TestQualityScorer._count_parametrize_cases(node)
                
                # Check for other data-driven patterns
                data_driven_count = TestQualityScorer._count_data_driven_cases(node)
                
                # Total effective test count (parameterized tests count as multiple)
                total_count = max(1, param_count + data_driven_count)
                
                test_info = {
                    'name': node.name,
                    'has_assertions': has_assertions,
                    'count': total_count,
                    'is_parameterized': param_count > 1,
                    'is_data_driven': data_driven_count > 1,
                    'parametrize_cases': param_count,
                    'data_driven_cases': data_driven_count
                }
                
                test_functions.append(test_info)
                logger.debug(f"Analyzed test function {node.name}: {test_info}")
        
        return test_functions
    
    @staticmethod
    def _count_parametrize_cases(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Count the number of test cases from pytest.mark.parametrize decorators."""
        total_cases = 0
        
        for decorator in func_node.decorator_list:
            if TestQualityScorer._is_parametrize_decorator(decorator):
                cases = TestQualityScorer._extract_parametrize_case_count(decorator)
                if cases > 0:
                    if total_cases == 0:
                        total_cases = cases
                    else:
                        total_cases *= cases  # Multiple parametrize decorators multiply
        
        return total_cases
    
    @staticmethod
    def _is_parametrize_decorator(decorator: ast.AST) -> bool:
        """Check if decorator is a pytest.mark.parametrize decorator."""
        # Handle @pytest.mark.parametrize
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                if (isinstance(decorator.func.value, ast.Attribute) and 
                    isinstance(decorator.func.value.value, ast.Name) and
                    decorator.func.value.value.id == "pytest" and
                    decorator.func.value.attr == "mark" and
                    decorator.func.attr == "parametrize"):
                    return True
        
        # Handle @parametrize (if imported directly)
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == "parametrize":
                return True
        
        # Handle @mark.parametrize (if mark imported directly)
        if isinstance(decorator, ast.Call):
            if (isinstance(decorator.func, ast.Attribute) and 
                isinstance(decorator.func.value, ast.Name) and
                decorator.func.value.id == "mark" and
                decorator.func.attr == "parametrize"):
                return True
        
        return False
    
    @staticmethod
    def _extract_parametrize_case_count(decorator: ast.Call) -> int:
        """Extract the number of test cases from a parametrize decorator."""
        if len(decorator.args) < 2:
            return 0
        
        # Second argument should be the test cases
        cases_arg = decorator.args[1]
        
        # Handle list of tuples: [(case1,), (case2,), ...]
        if isinstance(cases_arg, ast.List):
            return len(cases_arg.elts)
        
        # Handle list of lists: [[case1], [case2], ...]
        if isinstance(cases_arg, ast.List):
            return len(cases_arg.elts)
        
        # If we can't determine the count, assume it's meaningful (at least 2 cases)
        return 2
    
    @staticmethod
    def _count_data_driven_cases(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Count data-driven test cases from loops or data structures in the function."""
        loop_count = 0
        
        # Look for for loops that iterate over test data
        for node in ast.walk(func_node):
            if isinstance(node, ast.For):
                # Check if the loop variable is used in assertions
                loop_var = None
                if isinstance(node.target, ast.Name):
                    loop_var = node.target.id
                
                # Check if there are assertions inside the loop
                has_loop_assertions = any(
                    isinstance(n, ast.Assert) for n in ast.walk(node)
                )
                
                if has_loop_assertions and loop_var:
                    # Try to estimate the number of iterations
                    if isinstance(node.iter, ast.List):
                        loop_count += len(node.iter.elts)
                    elif isinstance(node.iter, ast.Call):
                        # range() call or similar
                        if (isinstance(node.iter.func, ast.Name) and 
                            node.iter.func.id == "range" and 
                            node.iter.args):
                            if isinstance(node.iter.args[0], ast.Constant):
                                loop_count += node.iter.args[0].value
                            else:
                                loop_count += 3  # Default assumption
                        else:
                            loop_count += 3  # Default for unknown iterables
                    else:
                        loop_count += 3  # Default assumption
        
        return loop_count
    
    def get_detailed_quality_metrics(self, tests_dir: str | Path) -> dict:
        """Return detailed quality metrics including parameterized test analysis."""
        logger = logging.getLogger(__name__)
        
        try:
            tests = Path(tests_dir)
            
            if not tests.exists() or not tests.is_dir():
                return {
                    'score': 0.0,
                    'total_functions': 0,
                    'total_test_cases': 0,
                    'functions_with_assertions': 0,
                    'parameterized_functions': 0,
                    'data_driven_functions': 0,
                    'error': 'Invalid tests directory'
                }
            
            test_files = list(tests.rglob("test_*.py"))
            if not test_files:
                return {
                    'score': 100.0,
                    'total_functions': 0,
                    'total_test_cases': 0,
                    'functions_with_assertions': 0,
                    'parameterized_functions': 0,
                    'data_driven_functions': 0,
                    'error': None
                }
            
            total_functions = 0
            total_test_cases = 0
            functions_with_assertions = 0
            parameterized_functions = 0
            data_driven_functions = 0
            
            for path in test_files:
                try:
                    tree, content = safe_parse_ast(path)
                    
                    test_functions = self._find_test_functions(tree)
                    for func_info in test_functions:
                        total_functions += 1
                        total_test_cases += func_info['count']
                        
                        if func_info['has_assertions']:
                            functions_with_assertions += 1
                        
                        if func_info['is_parameterized']:
                            parameterized_functions += 1
                        
                        if func_info['is_data_driven']:
                            data_driven_functions += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze test file {path}: {e}")
                    continue
            
            score = (functions_with_assertions / total_functions * 100) if total_functions > 0 else 100.0
            
            return {
                'score': score,
                'total_functions': total_functions,
                'total_test_cases': total_test_cases,
                'functions_with_assertions': functions_with_assertions,
                'parameterized_functions': parameterized_functions,
                'data_driven_functions': data_driven_functions,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed quality metrics for {tests_dir}: {e}")
            return {
                'score': 0.0,
                'total_functions': 0,
                'total_test_cases': 0,
                'functions_with_assertions': 0,
                'parameterized_functions': 0,
                'data_driven_functions': 0,
                'error': str(e)
            }
