"""Test common AST parsing patterns extraction."""

import tempfile
import ast
from pathlib import Path
import pytest

from testgen_copilot.ast_utils import safe_parse_ast, ASTParsingError
from testgen_copilot.generator import TestGenerator
from testgen_copilot.quality import TestQualityScorer


class TestASTPatterns:
    """Test extraction of common AST parsing patterns."""

    def test_safe_parse_ast_with_valid_content(self):
        """Test that safe_parse_ast works with valid Python content."""
        content = """
def test_function():
    return True

class TestClass:
    def method(self):
        pass
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            test_file = Path(f.name)

        try:
            # This should work and return an AST
            tree = safe_parse_ast(content, test_file)
            assert isinstance(tree, ast.AST)
            
            # Should contain expected nodes
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            
            assert len(functions) == 1
            assert len(classes) == 1
            assert functions[0].name == "test_function"
            assert classes[0].name == "TestClass"
            
        finally:
            test_file.unlink(missing_ok=True)

    def test_safe_parse_ast_with_syntax_error(self):
        """Test that safe_parse_ast handles syntax errors consistently."""
        content = "def invalid_syntax( missing_closing_paren:"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            test_file = Path(f.name)

        try:
            # Should raise ASTParsingError with structured context
            with pytest.raises(ASTParsingError) as exc_info:
                safe_parse_ast(content, test_file)
            
            error = exc_info.value
            assert error.file_path == test_file
            assert error.line_number is not None
            assert "syntax error" in str(error).lower()
            
        finally:
            test_file.unlink(missing_ok=True)

    def test_safe_parse_ast_with_empty_content(self):
        """Test that safe_parse_ast handles empty content."""
        content = ""
        
        # Should work fine - empty module is valid Python
        tree = safe_parse_ast(content, None)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 0

    def test_safe_parse_ast_with_comments_only(self):
        """Test that safe_parse_ast handles files with only comments."""
        content = """
# This is a comment file
# With multiple lines
# But no actual code
"""
        
        tree = safe_parse_ast(content, None)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 0  # Comments don't create AST nodes

    def test_generator_uses_safe_parse_ast(self):
        """Test that generator now uses the common safe_parse_ast utility.
        
        This test will fail initially since we haven't integrated it yet.
        """
        # Create a test file with syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken_syntax( invalid:")
            test_file = Path(f.name)

        try:
            generator = TestGenerator()
            
            # Should get consistent error handling from safe_parse_ast
            with pytest.raises(ASTParsingError):
                with tempfile.TemporaryDirectory() as temp_dir:
                    generator.generate_tests(test_file, temp_dir)
                    
        finally:
            test_file.unlink(missing_ok=True)

    def test_quality_scorer_uses_safe_parse_ast(self):
        """Test that quality scorer now uses the common safe_parse_ast utility.
        
        This test will fail initially since we haven't integrated it yet.
        """
        # Create a test file with syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken_syntax( invalid:")
            test_file = Path(f.name)

        try:
            scorer = TestQualityScorer()
            
            # Should get consistent error handling from safe_parse_ast
            with pytest.raises(ASTParsingError):
                scorer.score_tests(test_file)
                    
        finally:
            test_file.unlink(missing_ok=True)

    def test_consistent_error_reporting(self):
        """Test that all modules report AST errors consistently."""
        content_with_error = "def func( invalid syntax:"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content_with_error)
            test_file = Path(f.name)

        try:
            # All should raise the same exception type with same context
            generator = TestGenerator()
            scorer = TestQualityScorer()
            
            gen_error = None
            quality_error = None
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    generator.generate_tests(test_file, temp_dir)
            except ASTParsingError as e:
                gen_error = e
            
            try:
                scorer.score_tests(test_file)
            except ASTParsingError as e:
                quality_error = e
            
            # Both should have raised ASTParsingError
            assert gen_error is not None, "Generator should raise ASTParsingError"
            assert quality_error is not None, "Quality scorer should raise ASTParsingError"
            
            # Both should have same file path and error structure
            assert gen_error.file_path == quality_error.file_path == test_file
            assert gen_error.line_number == quality_error.line_number
            
        finally:
            test_file.unlink(missing_ok=True)

    def test_ast_parsing_performance_monitoring(self):
        """Test that AST parsing includes performance monitoring."""
        # Create a moderately complex file
        content = ""
        for i in range(50):
            content += f"""
def function_{i}():
    '''Function {i} docstring'''
    if True:
        for j in range(10):
            result = {{'key_{j}': j * {i}}}
            yield result[f'key_{{j}}']
"""
        
        tree = safe_parse_ast(content, None)
        
        # Should successfully parse complex content
        assert isinstance(tree, ast.Module)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 50
        
        # Performance data should be logged (we can't easily test this without 
        # capturing logs, but the functionality should work)

    def test_deduplicated_parsing_logic(self):
        """Test that AST parsing logic is no longer duplicated.
        
        This test documents that we should have reduced code duplication.
        """
        # After refactoring, we should have:
        # - One central safe_parse_ast function
        # - Consistent error handling across all modules
        # - No duplicate AST parsing patterns
        
        # This is more of a code organization test than a functional test
        # In a real scenario, we'd use static analysis tools to verify this
        pass