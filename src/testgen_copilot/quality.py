"""Utilities to score the quality of test files."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from .file_utils import safe_read_file, FileSizeError, safe_parse_ast as file_safe_parse_ast
from .logging_config import get_quality_logger
from .ast_utils import safe_parse_ast as ast_safe_parse_ast, ASTParsingError
from .cache import cached_operation, analysis_cache


class TestQualityScorer:
    """Estimate quality of tests based on presence of assertions and advanced patterns."""

    def score(self, tests_dir: str | Path) -> float:
        """Return percentage of test functions containing ``assert`` statements."""
        logger = get_quality_logger()
        
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
                    result = file_safe_parse_ast(path, raise_on_syntax_error=False)
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
        logger = get_quality_logger()
        
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
                    result = file_safe_parse_ast(path, raise_on_syntax_error=False)
                    if result is None:
                        # safe_parse_ast already logged the syntax error
                        continue
                    
                    try:
                        content = safe_read_file(path)
                    except (FileNotFoundError, PermissionError, ValueError, FileSizeError, OSError) as e:
                        logger.warning(f"Cannot read test file {path}: {e}")
                        continue
                    
                    try:
                        tree = ast_safe_parse_ast(content, path)
                    except ASTParsingError as e:
                        logger.warning("Skipping test file due to parsing error", {
                            "file_path": str(path),
                            "error_message": str(e)
                        })
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
        logger = get_quality_logger()
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
                
                # Check for fixture usage
                fixture_info = TestQualityScorer._analyze_fixture_usage(node)
                
                test_info = {
                    'name': node.name,
                    'has_assertions': has_assertions,
                    'count': total_count,
                    'is_parameterized': param_count > 1,
                    'is_data_driven': data_driven_count > 1,
                    'parametrize_cases': param_count,
                    'data_driven_cases': data_driven_count,
                    'fixtures_used': fixture_info['used'],
                    'missing_fixtures': fixture_info['missing'],
                    'fixture_count': len(fixture_info['used'])
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
        logger = get_quality_logger()
        
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
            functions_with_fixtures = 0
            missing_fixtures_count = 0
            all_missing_fixtures = []
            
            for path in test_files:
                try:
                    result = file_safe_parse_ast(path, raise_on_syntax_error=False)
                    if result is None:
                        continue
                    tree, content = result
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
                        
                        if func_info['fixture_count'] > 0:
                            functions_with_fixtures += 1
                        
                        if func_info['missing_fixtures']:
                            missing_fixtures_count += len(func_info['missing_fixtures'])
                            all_missing_fixtures.extend(func_info['missing_fixtures'])
                            
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
                'functions_with_fixtures': functions_with_fixtures,
                'missing_fixtures_count': missing_fixtures_count,
                'missing_fixtures': all_missing_fixtures,
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
                'functions_with_fixtures': 0,
                'missing_fixtures_count': 0,
                'missing_fixtures': [],
                'error': str(e)
            }
    
    @staticmethod
    def _analyze_fixture_usage(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict:
        """Analyze fixture usage in a test function."""
        used_fixtures = set()
        missing_fixtures = []
        
        # Extract fixture parameters from function signature
        fixture_params = []
        for arg in func_node.args.args:
            if arg.arg not in ['self', 'cls']:  # Exclude method parameters
                fixture_params.append(arg.arg)
        
        # Common fixture patterns to look for
        common_fixtures = {
            'tmpdir', 'tmp_path', 'monkeypatch', 'capsys', 'capfd', 
            'client', 'app', 'db', 'session', 'mock', 'mocker',
            'request', 'cache', 'settings'
        }
        
        # Analyze function body for fixture usage patterns
        for node in ast.walk(func_node):
            # Look for variable usage that suggests fixtures
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                var_name = node.id
                
                # Check if it matches fixture parameters
                if var_name in fixture_params:
                    used_fixtures.add(var_name)
                
                # Check for common fixture patterns
                elif var_name in common_fixtures:
                    used_fixtures.add(var_name)
            
            # Look for attribute access that suggests fixture usage
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    base_name = node.value.id
                    if base_name in fixture_params:
                        used_fixtures.add(base_name)
        
        # Identify potentially missing fixtures based on code patterns
        body_code = ast.unparse(func_node) if hasattr(ast, 'unparse') else str(func_node)
        
        # Check for patterns that suggest missing fixtures
        missing_patterns = {
            'tmpdir': ['tempfile.', 'temp_dir', 'temporary'],
            'monkeypatch': ['setattr(', 'delattr(', 'setenv('],
            'capsys': ['print(', 'sys.stdout', 'captured'],
            'mock': ['Mock(', 'patch(', 'mock.'],
            'client': ['requests.', 'http', 'response.'],
            'db': ['database', 'query', 'session.']
        }
        
        for fixture_name, patterns in missing_patterns.items():
            if fixture_name not in used_fixtures and fixture_name not in fixture_params:
                if any(pattern in body_code for pattern in patterns):
                    missing_fixtures.append({
                        'fixture': fixture_name,
                        'reason': f'Code contains patterns suggesting {fixture_name} usage',
                        'patterns_found': [p for p in patterns if p in body_code]
                    })
        
        # Check for hardcoded paths/values that could use fixtures
        if 'tmp_path' not in used_fixtures and 'tmp_path' not in fixture_params:
            if any(pattern in body_code for pattern in ['/tmp/', '/temp/', 'C:\\temp']):
                missing_fixtures.append({
                    'fixture': 'tmp_path',
                    'reason': 'Hardcoded temporary paths found',
                    'patterns_found': ['hardcoded paths']
                })
        
        return {
            'used': list(used_fixtures),
            'missing': missing_fixtures
        }
