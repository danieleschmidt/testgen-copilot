"""
Performance benchmark tests for core TestGen functionality.
"""
import pytest
from pathlib import Path
import tempfile
import time

from testgen_copilot.core import TestGenOrchestrator
from testgen_copilot.generator import TestGenerator
from testgen_copilot.ast_utils import ASTParsingError


@pytest.fixture
def sample_python_file():
    """Create a sample Python file for benchmarking."""
    content = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
        
    def get_history(self):
        return self.history.copy()
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.mark.benchmark
@pytest.mark.performance
def test_ast_parsing_performance(benchmark, sample_python_file):
    """Benchmark AST parsing performance."""
    from testgen_copilot.file_utils import safe_parse_ast
    
    def parse_file():
        return safe_parse_ast(sample_python_file)
    
    result = benchmark(parse_file)
    assert result is not None
    
    # Cleanup
    sample_python_file.unlink()


@pytest.mark.benchmark
@pytest.mark.performance
def test_test_generation_performance(benchmark, sample_python_file):
    """Benchmark test generation performance."""
    import tempfile
    generator = TestGenerator()
    
    def generate_tests():
        with tempfile.TemporaryDirectory() as temp_dir:
            return generator.generate_tests(sample_python_file, temp_dir)
    
    result = benchmark(generate_tests)
    assert result is not None
    assert result.exists()
    
    # Cleanup
    sample_python_file.unlink()


@pytest.mark.benchmark
@pytest.mark.performance
def test_large_file_processing(benchmark):
    """Benchmark processing of a large Python file."""
    
    # Generate a large Python file
    large_content = '''
def function_{i}(x):
    """Function number {i}."""
    return x * {i}

class Class_{i}:
    """Class number {i}."""
    
    def method_{i}(self, x):
        return x + {i}
        
    def another_method_{i}(self, x, y):
        return x * y + {i}
'''
    
    # Create 100 functions and classes
    content = "\n".join(large_content.format(i=i) for i in range(100))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        large_file = Path(f.name)
    
    orchestrator = TestGenOrchestrator(repo_path=Path("."))
    
    def process_large_file():
        import asyncio
        return asyncio.run(orchestrator.process_file(large_file, "/tmp/test_output"))
    
    result = benchmark(process_large_file)
    assert result is not None
    
    # Cleanup
    large_file.unlink()


@pytest.mark.performance
@pytest.mark.slow
def test_memory_usage_tracking():
    """Test memory usage during processing."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process multiple files
    orchestrator = TestGenOrchestrator(repo_path=Path("."))
    
    for i in range(10):
        content = f'''
def test_function_{i}(x):
    return x * {i}

class TestClass_{i}:
    def method(self, x):
        return x + {i}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)
        
        import asyncio
        asyncio.run(orchestrator.process_file(temp_file, "/tmp/test_output"))
        temp_file.unlink()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 100MB for this test)
    assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"


@pytest.mark.performance
def test_concurrent_processing_performance():
    """Test performance of concurrent file processing."""
    import concurrent.futures
    import threading
    
    # Create multiple test files
    files = []
    for i in range(5):
        content = f'''
def function_{i}(x):
    return x * {i}

class Class_{i}:
    def method(self, x):
        return x + {i}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            files.append(Path(f.name))
    
    orchestrator = TestGenOrchestrator(repo_path=Path("."))
    
    # Test sequential processing  
    start_time = time.time()
    for file_path in files:
        import asyncio
        asyncio.run(orchestrator.process_file(file_path, "/tmp/test_output"))
    sequential_time = time.time() - start_time
    
    # Test concurrent processing
    import asyncio
    async def process_files():
        results = await orchestrator.process_project(Path("/tmp"), Path("/tmp/test_output"))
        return results
    
    start_time = time.time()
    asyncio.run(process_files())
    concurrent_time = time.time() - start_time
    
    # Concurrent processing should be faster (or at least not significantly slower)
    speedup_ratio = sequential_time / concurrent_time
    assert speedup_ratio > 0.8, f"Concurrent processing is too slow (speedup: {speedup_ratio:.2f}x)"
    
    # Cleanup
    for file_path in files:
        file_path.unlink()


@pytest.mark.performance
def test_caching_effectiveness():
    """Test that caching improves performance."""
    content = '''
def test_function(x):
    return x * 2

class TestClass:
    def method(self, x):
        return x + 1
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        test_file = Path(f.name)
    
    orchestrator = TestGenOrchestrator(repo_path=Path("."))
    
    # First analysis (cache miss)
    start_time = time.time()
    import asyncio
    result1 = asyncio.run(orchestrator.process_file(test_file, "/tmp/test_output"))
    first_analysis_time = time.time() - start_time
    
    # Second analysis (cache hit)
    start_time = time.time()
    result2 = asyncio.run(orchestrator.process_file(test_file, "/tmp/test_output"))
    second_analysis_time = time.time() - start_time
    
    # Results should have similar structure
    assert result1.status == result2.status
    
    # Second analysis should be significantly faster
    speedup = first_analysis_time / second_analysis_time
    assert speedup > 2, f"Caching not effective enough (speedup: {speedup:.2f}x)"
    
    # Cleanup
    test_file.unlink()