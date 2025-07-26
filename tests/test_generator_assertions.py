"""Test proper assertion generation in TestGenerator."""

import tempfile
from pathlib import Path

import pytest

from testgen_copilot.generator import GenerationConfig, TestGenerator


class TestGeneratorAssertions:
    """Test that generated tests include proper assertions instead of TODOs."""

    def test_python_function_generates_proper_assertion(self):
        """Test that Python function generates assertion based on return type."""
        # Create sample Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def calculate_sum(a, b):
    \"\"\"Return sum of two numbers.\"\"\"
    return a + b

def get_greeting(name):
    \"\"\"Return greeting string.\"\"\"
    return f"Hello, {name}!"

def is_even(number):
    \"\"\"Check if number is even.\"\"\"
    return number % 2 == 0
""")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = TestGenerator()
            output_path = generator.generate_tests(source_path, temp_dir)
            
            content = output_path.read_text()
            
            # Should not contain TODO placeholders
            assert "# TODO:" not in content
            assert "// TODO:" not in content
            
            # Should contain proper assertions
            assert "assert result ==" in content or "assert isinstance" in content
            
        source_path.unlink()

    def test_javascript_function_generates_proper_expectation(self):
        """Test that JavaScript function generates expect() instead of TODO."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write("""
function add(a, b) {
    return a + b;
}

const multiply = (x, y) => x * y;
""")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(language="javascript")
            generator = TestGenerator(config)
            output_path = generator.generate_tests(source_path, temp_dir)
            
            content = output_path.read_text()
            
            # Should not contain TODO placeholders
            assert "// TODO:" not in content
            
            # Should contain proper expectations
            assert "expect(" in content
            
        source_path.unlink()

    def test_java_method_generates_proper_assertion(self):
        """Test that Java method generates Assert instead of TODO."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write("""
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public String getMessage() {
        return "Hello World";
    }
}
""")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(language="java")
            generator = TestGenerator(config)
            output_path = generator.generate_tests(source_path, temp_dir)
            
            content = output_path.read_text()
            
            # Should not contain TODO placeholders
            assert "// TODO:" not in content
            
            # Should contain proper assertions
            assert "assert" in content.lower() and ("assertNotNull" in content or "assertTrue" in content or "assertEquals" in content)
            
        source_path.unlink()

    def test_edge_case_test_has_proper_assertion(self):
        """Test that edge case tests also have proper assertions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def divide(a, b):
    \"\"\"Divide two numbers.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(include_edge_cases=True)
            generator = TestGenerator(config)
            output_path = generator.generate_tests(source_path, temp_dir)
            
            content = output_path.read_text()
            
            # Should not contain any TODO placeholders
            assert "# TODO:" not in content
            
            # Should have edge case test with assertion
            assert "test_divide_edge_case" in content
            assert "assert" in content
            
        source_path.unlink()

    def test_integration_test_has_assertions(self):
        """Test that integration tests call functions without TODOs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def process_data(data):
    return data.upper()

def validate_input(input_str):
    return len(input_str) > 0
""")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerationConfig(include_integration_tests=True)
            generator = TestGenerator(config)
            output_path = generator.generate_tests(source_path, temp_dir)
            
            content = output_path.read_text()
            
            # Should have integration test
            assert "_integration" in content
            # Should call both functions
            assert "process_data(" in content
            assert "validate_input(" in content
            
        source_path.unlink()