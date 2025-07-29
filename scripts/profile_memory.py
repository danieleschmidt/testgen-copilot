#!/usr/bin/env python3
"""Memory profiling script for TestGen Copilot."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_profiler import profile
from testgen_copilot.cli import main
from testgen_copilot.generator import TestGenerator
from testgen_copilot.core import TestGenCore


@profile
def profile_test_generation():
    """Profile memory usage during test generation."""
    generator = TestGenerator(language="python")
    
    # Simulate test generation on sample code
    sample_code = """
def calculate_discount(price, discount_percent):
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""
    
    # Generate tests multiple times to see memory patterns
    for i in range(10):
        tests = generator.generate_from_code(sample_code)
        print(f"Generated test iteration {i + 1}: {len(tests)} lines")


@profile
def profile_core_operations():
    """Profile core TestGen operations."""
    core = TestGenCore()
    
    # Profile various core operations
    for i in range(5):
        result = core.analyze_code("sample_code.py")
        print(f"Analysis iteration {i + 1}: {len(result) if result else 0} items")


if __name__ == "__main__":
    print("Starting memory profiling...")
    print("\n=== Test Generation Profiling ===")
    profile_test_generation()
    
    print("\n=== Core Operations Profiling ===")
    profile_core_operations()
    
    print("\nMemory profiling completed.")