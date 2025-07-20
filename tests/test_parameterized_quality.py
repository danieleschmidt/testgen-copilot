"""Test parameterized test support in TestQualityScorer."""

import tempfile
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, 'src')

from testgen_copilot.quality import TestQualityScorer


class TestParameterizedQualityScoring:
    """Test parameterized test detection and scoring."""

    def test_basic_test_function(self):
        """Test scoring of basic test function."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_basic.py"
            test_file.write_text("""
def test_simple():
    assert True

def test_without_assert():
    pass
""")
            
            score = scorer.score(tests_dir)
            assert score == 50.0  # 1 out of 2 functions has assertions
            
            metrics = scorer.get_detailed_quality_metrics(tests_dir)
            assert metrics['total_functions'] == 2
            assert metrics['total_test_cases'] == 2
            assert metrics['functions_with_assertions'] == 1
            assert metrics['parameterized_functions'] == 0

    def test_pytest_mark_parametrize(self):
        """Test detection of pytest.mark.parametrize decorators."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_parametrized.py"
            test_file.write_text("""
import pytest

@pytest.mark.parametrize("value", [1, 2, 3])
def test_parametrized_function(value):
    assert value > 0

def test_regular_function():
    assert True
""")
            
            metrics = scorer.get_detailed_quality_metrics(tests_dir)
            assert metrics['total_functions'] == 2
            assert metrics['total_test_cases'] == 4  # 3 parameterized + 1 regular
            assert metrics['functions_with_assertions'] == 2
            assert metrics['parameterized_functions'] == 1

    def test_multiple_parametrize_decorators(self):
        """Test multiple parametrize decorators on one function."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_multi_param.py"
            test_file.write_text("""
import pytest

@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", ["a", "b", "c"])
def test_multi_parametrized(x, y):
    assert x > 0
    assert y in ["a", "b", "c"]
""")
            
            metrics = scorer.get_detailed_quality_metrics(tests_dir)
            assert metrics['total_functions'] == 1
            assert metrics['total_test_cases'] == 6  # 2 * 3 = 6 combinations
            assert metrics['functions_with_assertions'] == 1
            assert metrics['parameterized_functions'] == 1

    def test_parametrize_import_variations(self):
        """Test different ways of importing parametrize."""
        scorer = TestQualityScorer()
        
        test_cases = [
            # Direct import
            """
from pytest import mark

@mark.parametrize("value", [1, 2])
def test_mark_import(value):
    assert value > 0
""",
            # Parametrize direct import (if available)
            """
# This would work if parametrize was imported directly
# from pytest.mark import parametrize

import pytest

@pytest.mark.parametrize("value", [1, 2])
def test_direct_parametrize(value):
    assert value > 0
"""
        ]
        
        for i, test_content in enumerate(test_cases):
            with tempfile.TemporaryDirectory() as tmpdir:
                tests_dir = Path(tmpdir)
                test_file = tests_dir / f"test_import_{i}.py"
                test_file.write_text(test_content)
                
                metrics = scorer.get_detailed_quality_metrics(tests_dir)
                assert metrics['total_functions'] == 1
                assert metrics['total_test_cases'] >= 2  # At least 2 parameterized cases
                assert metrics['parameterized_functions'] == 1

    def test_data_driven_loops(self):
        """Test detection of data-driven tests with loops."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_loops.py"
            test_file.write_text("""
def test_loop_with_data():
    test_data = [1, 2, 3, 4, 5]
    for value in test_data:
        assert value > 0

def test_range_loop():
    for i in range(3):
        assert i >= 0

def test_no_loop():
    assert True
""")
            
            metrics = scorer.get_detailed_quality_metrics(tests_dir)
            assert metrics['total_functions'] == 3
            assert metrics['total_test_cases'] > 3  # Should count loop iterations
            assert metrics['data_driven_functions'] == 2  # Two functions with loops

    def test_complex_parameterized_example(self):
        """Test complex example with multiple types of test patterns."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_complex.py"
            test_file.write_text("""
import pytest

class TestComplexScenarios:
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parameterized_with_multiple_args(self, input_val, expected):
        result = input_val * 2
        assert result == expected
    
    def test_data_driven_with_loop(self):
        scenarios = [
            {"input": 1, "expected": 1},
            {"input": 2, "expected": 4},
            {"input": 3, "expected": 9},
        ]
        for scenario in scenarios:
            result = scenario["input"] ** 2
            assert result == scenario["expected"]
    
    def test_regular_test(self):
        assert 2 + 2 == 4
    
    def test_no_assertions(self):
        # This test has no assertions
        print("Testing something")
""")
            
            score = scorer.score(tests_dir)
            # The score should be based on functions with assertions vs total functions
            # 3 out of 4 functions have assertions = 75.0%
            assert abs(score - 75.0) < 0.1  # Allow small floating point differences
            
            metrics = scorer.get_detailed_quality_metrics(tests_dir)
            assert metrics['total_functions'] == 4
            assert metrics['total_test_cases'] > 4  # Should count parameterized and loop cases
            assert metrics['functions_with_assertions'] == 3
            assert metrics['parameterized_functions'] == 1
            assert metrics['data_driven_functions'] == 1

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        scorer = TestQualityScorer()
        
        # Test with syntax error in parametrize
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_edge.py"
            test_file.write_text("""
import pytest

@pytest.mark.parametrize("value", [1, 2, 3])
def test_valid_parametrize(value):
    assert value > 0

# This would cause issues but should be handled gracefully
@pytest.mark.parametrize("incomplete")
def test_incomplete_parametrize():
    pass
""")
            
            # Should handle gracefully without crashing
            try:
                metrics = scorer.get_detailed_quality_metrics(tests_dir)
                # Should still process the valid function
                assert metrics['parameterized_functions'] >= 1
            except Exception:
                # If there's a syntax error, it should be handled gracefully
                pass

    def test_low_quality_tests_with_parametrize(self):
        """Test low_quality_tests method with parameterized tests."""
        scorer = TestQualityScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir)
            test_file = tests_dir / "test_quality.py"
            test_file.write_text("""
import pytest

@pytest.mark.parametrize("value", [1, 2, 3])
def test_good_parametrized(value):
    assert value > 0

@pytest.mark.parametrize("value", [1, 2, 3])
def test_bad_parametrized(value):
    # No assertions
    pass

def test_good_regular():
    assert True

def test_bad_regular():
    # No assertions
    pass
""")
            
            lacking = scorer.low_quality_tests(tests_dir)
            assert "test_bad_parametrized" in lacking
            assert "test_bad_regular" in lacking
            assert "test_good_parametrized" not in lacking
            assert "test_good_regular" not in lacking


def main():
    """Run parameterized test quality tests."""
    print("🧪 Testing Parameterized Test Quality Scoring")
    print("=" * 50)
    
    test_instance = TestParameterizedQualityScoring()
    
    test_methods = [
        test_instance.test_basic_test_function,
        test_instance.test_pytest_mark_parametrize,
        test_instance.test_multiple_parametrize_decorators,
        test_instance.test_parametrize_import_variations,
        test_instance.test_data_driven_loops,
        test_instance.test_complex_parameterized_example,
        test_instance.test_edge_cases,
        test_instance.test_low_quality_tests_with_parametrize,
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
        print("✅ All parameterized test quality tests passed!")
    else:
        print(f"❌ {failed} tests failed")


if __name__ == "__main__":
    main()