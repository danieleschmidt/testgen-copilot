"""Test centralized AST parsing utility consolidation."""

import ast
import sys
import tempfile
from pathlib import Path
from enum import Enum
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, 'src')

from testgen_copilot.file_utils import safe_parse_ast, SyntaxErrorStrategy


class TestASTParsingConsolidation:
    """Test centralized AST parsing functionality."""

    def test_safe_parse_ast_basic_functionality(self):
        """Test basic AST parsing with valid Python code."""
        test_code = """
def test_function():
    return "hello world"

class TestClass:
    def method(self):
        pass
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(test_code)
            f.flush()
            
            try:
                # Should return tuple of (ast.AST, content)
                tree, content = safe_parse_ast(f.name)
                
                assert isinstance(tree, ast.AST), "Should return AST object"
                assert isinstance(content, str), "Should return file content"
                assert "test_function" in content, "Content should contain source code"
                
                # Check AST structure
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
                
                assert len(functions) == 1, "Should find one function"
                assert len(classes) == 1, "Should find one class"
                assert functions[0].name == "test_function", "Should parse function name correctly"
                
            finally:
                Path(f.name).unlink()

    def test_safe_parse_ast_with_content_parameter(self):
        """Test AST parsing when content is provided directly."""
        test_code = "def example(): pass"
        test_path = "dummy.py"  # Path doesn't need to exist when content is provided
        
        tree, content = safe_parse_ast(test_path, content=test_code)
        
        assert isinstance(tree, ast.AST), "Should return AST object"
        assert content == test_code, "Should return provided content"
        
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1, "Should find the function"
        assert functions[0].name == "example", "Should parse function name correctly"

    def test_syntax_error_handling_strategies(self):
        """Test different syntax error handling strategies."""
        invalid_code = "def invalid_syntax( # missing closing parenthesis"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(invalid_code)
            f.flush()
            
            try:
                # Test RAISE strategy (default)
                try:
                    safe_parse_ast(f.name, raise_on_syntax_error=True)
                    assert False, "Should have raised SyntaxError"
                except SyntaxError as e:
                    assert "invalid_syntax" in str(e) or "syntax error" in str(e).lower()
                
                # Test returning None for invalid syntax when not raising
                result = safe_parse_ast(f.name, raise_on_syntax_error=False)
                assert result is None, "Should return None for invalid syntax when not raising"
                
            finally:
                Path(f.name).unlink()

    def test_syntax_error_strategy_enum(self):
        """Test that SyntaxErrorStrategy enum is defined and usable."""
        # Test that enum exists and has expected values
        assert hasattr(SyntaxErrorStrategy, 'RAISE'), "Should have RAISE strategy"
        assert hasattr(SyntaxErrorStrategy, 'WARN_AND_SKIP'), "Should have WARN_AND_SKIP strategy"
        assert hasattr(SyntaxErrorStrategy, 'RETURN_ERROR'), "Should have RETURN_ERROR strategy"
        
        # Test enum values
        assert SyntaxErrorStrategy.RAISE.value == "raise"
        assert SyntaxErrorStrategy.WARN_AND_SKIP.value == "warn_and_skip"
        assert SyntaxErrorStrategy.RETURN_ERROR.value == "return_error"

    def test_file_size_limits_integration(self):
        """Test that safe_parse_ast respects file size limits."""
        # Create a large file that exceeds default limits
        large_content = "# " + "x" * (11 * 1024 * 1024)  # 11MB
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(large_content)
            f.flush()
            
            try:
                # Should raise FileSizeError due to size limit
                from testgen_copilot.file_utils import FileSizeError
                try:
                    safe_parse_ast(f.name, max_size_mb=10)
                    assert False, "Should have raised FileSizeError"
                except FileSizeError:
                    pass  # Expected
                    
            finally:
                Path(f.name).unlink()

    def test_timeout_integration(self):
        """Test that safe_parse_ast can use timeout protection."""
        # Simple valid code should parse quickly even with short timeout
        simple_code = "def simple(): pass"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(simple_code)
            f.flush()
            
            try:
                # Should complete successfully with reasonable timeout
                tree, content = safe_parse_ast(f.name, timeout_seconds=5)
                assert isinstance(tree, ast.AST), "Should parse successfully with timeout"
                
            finally:
                Path(f.name).unlink()

    def test_error_context_and_logging(self):
        """Test that safe_parse_ast provides detailed error context."""
        invalid_code = """
def function_one():
    pass

def function_two(:  # syntax error on this line
    pass
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(invalid_code)
            f.flush()
            
            try:
                try:
                    safe_parse_ast(f.name)
                    assert False, "Should have raised SyntaxError"
                except SyntaxError as e:
                    # Error should include file context
                    error_msg = str(e)
                    assert f.name in error_msg or "syntax error" in error_msg.lower()
                    assert hasattr(e, 'lineno'), "Should preserve line number information"
                    assert e.lineno is not None, "Line number should be available"
                    
            finally:
                Path(f.name).unlink()

    def test_integration_with_existing_modules(self):
        """Test that safe_parse_ast can replace existing patterns."""
        # This test verifies the new utility works for common use cases
        test_code = """
import os
import sys

def function_with_docstring():
    '''This is a test function.'''
    return 42

class ExampleClass:
    def method(self):
        pass

if __name__ == "__main__":
    print("Hello, world!")
"""
        
        tree, content = safe_parse_ast("test.py", content=test_code)
        
        # Verify we can extract the same information that existing modules need
        
        # Function extraction (like generator.py)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1
        assert functions[0].name == "function_with_docstring"
        
        # Import analysis (like security.py)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert len(imports) == 2  # os and sys
        
        # Class analysis (like quality.py)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert len(classes) == 1
        assert classes[0].name == "ExampleClass"


def main():
    """Run AST parsing consolidation tests."""
    print("🧪 Testing AST Parsing Consolidation")
    print("=" * 50)
    
    test_instance = TestASTParsingConsolidation()
    
    test_methods = [
        test_instance.test_safe_parse_ast_basic_functionality,
        test_instance.test_safe_parse_ast_with_content_parameter,
        test_instance.test_syntax_error_handling_strategies,
        test_instance.test_syntax_error_strategy_enum,
        test_instance.test_file_size_limits_integration,
        test_instance.test_timeout_integration,
        test_instance.test_error_context_and_logging,
        test_instance.test_integration_with_existing_modules,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All AST parsing consolidation tests passed!")
    else:
        print(f"❌ {failed} tests failed - AST parsing consolidation implementation needed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)