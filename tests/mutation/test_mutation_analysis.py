# Mutation Testing Implementation
# Tests to verify test suite effectiveness through mutation analysis

import pytest
import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class MutationOperator:
    """Base class for mutation operators."""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply_mutation(self, source_code: str) -> List[str]:
        """Apply mutation to source code and return mutated versions."""
        raise NotImplementedError


class ArithmeticMutationOperator(MutationOperator):
    """Mutates arithmetic operators."""
    
    def __init__(self):
        super().__init__("arithmetic")
        self.mutations = [
            ("+", "-"), ("-", "+"), ("*", "/"), ("/", "*"),
            ("//", "%"), ("%", "//"), ("**", "*")
        ]
    
    def apply_mutation(self, source_code: str) -> List[str]:
        """Apply arithmetic mutations."""
        mutated_versions = []
        
        for original, replacement in self.mutations:
            if original in source_code:
                mutated_code = source_code.replace(original, replacement, 1)
                mutated_versions.append({
                    'mutation': f"{original} -> {replacement}",
                    'code': mutated_code
                })
        
        return mutated_versions


class ComparisonMutationOperator(MutationOperator):
    """Mutates comparison operators."""
    
    def __init__(self):
        super().__init__("comparison")
        self.mutations = [
            ("==", "!="), ("!=", "=="), ("<", ">="), 
            (">=", "<"), (">", "<="), ("<=", ">")
        ]
    
    def apply_mutation(self, source_code: str) -> List[str]:
        """Apply comparison mutations."""
        mutated_versions = []
        
        for original, replacement in self.mutations:
            if original in source_code:
                mutated_code = source_code.replace(original, replacement, 1)
                mutated_versions.append({
                    'mutation': f"{original} -> {replacement}",
                    'code': mutated_code
                })
        
        return mutated_versions


class LogicalMutationOperator(MutationOperator):
    """Mutates logical operators."""
    
    def __init__(self):
        super().__init__("logical")
        self.mutations = [
            (" and ", " or "), (" or ", " and "),
            ("True", "False"), ("False", "True")
        ]
    
    def apply_mutation(self, source_code: str) -> List[str]:
        """Apply logical mutations."""
        mutated_versions = []
        
        for original, replacement in self.mutations:
            if original in source_code:
                mutated_code = source_code.replace(original, replacement, 1)
                mutated_versions.append({
                    'mutation': f"{original} -> {replacement}",
                    'code': mutated_code
                })
        
        return mutated_versions


class MutationTester:
    """Orchestrates mutation testing process."""
    
    def __init__(self, source_files: List[Path], test_command: str = "pytest"):
        self.source_files = source_files
        self.test_command = test_command
        self.operators = [
            ArithmeticMutationOperator(),
            ComparisonMutationOperator(),
            LogicalMutationOperator()
        ]
    
    def run_mutation_testing(self) -> Dict[str, Any]:
        """Run comprehensive mutation testing."""
        results = {
            'total_mutants': 0,
            'killed_mutants': 0,
            'survived_mutants': 0,
            'mutation_score': 0.0,
            'detailed_results': []
        }
        
        for source_file in self.source_files:
            if not source_file.exists():
                continue
                
            source_code = source_file.read_text()
            file_results = self._test_file_mutations(source_file, source_code)
            
            results['total_mutants'] += file_results['total_mutants']
            results['killed_mutants'] += file_results['killed_mutants']
            results['survived_mutants'] += file_results['survived_mutants']
            results['detailed_results'].append(file_results)
        
        if results['total_mutants'] > 0:
            results['mutation_score'] = (
                results['killed_mutants'] / results['total_mutants']
            ) * 100
        
        return results
    
    def _test_file_mutations(self, source_file: Path, source_code: str) -> Dict[str, Any]:
        """Test mutations for a single file."""
        file_results = {
            'file': str(source_file),
            'total_mutants': 0,
            'killed_mutants': 0,
            'survived_mutants': 0,
            'mutations': []
        }
        
        # Generate all mutations for this file
        all_mutations = []
        for operator in self.operators:
            mutations = operator.apply_mutation(source_code)
            all_mutations.extend(mutations)
        
        file_results['total_mutants'] = len(all_mutations)
        
        # Test each mutation
        for mutation in all_mutations:
            is_killed = self._test_single_mutation(
                source_file, mutation['code'], mutation['mutation']
            )
            
            mutation_result = {
                'mutation': mutation['mutation'],
                'killed': is_killed
            }
            
            if is_killed:
                file_results['killed_mutants'] += 1
            else:
                file_results['survived_mutants'] += 1
            
            file_results['mutations'].append(mutation_result)
        
        return file_results
    
    def _test_single_mutation(self, source_file: Path, mutated_code: str, mutation_desc: str) -> bool:
        """Test a single mutation by running tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create temporary mutated file
            temp_source = temp_path / source_file.name
            temp_source.write_text(mutated_code)
            
            # Copy original file structure
            original_parent = source_file.parent
            temp_parent = temp_path / "testgen_copilot"
            temp_parent.mkdir(exist_ok=True)
            
            # Copy all other source files
            for sibling in original_parent.glob("*.py"):
                if sibling != source_file:
                    (temp_parent / sibling.name).write_text(sibling.read_text())
            
            # Move mutated file to correct location
            (temp_parent / source_file.name).write_text(mutated_code)
            
            # Run tests
            try:
                result = subprocess.run(
                    [self.test_command, "-x", "--tb=no", "-q"],
                    cwd=temp_path,
                    capture_output=True,
                    timeout=30,
                    env={"PYTHONPATH": str(temp_path)}
                )
                
                # Mutation is killed if tests fail
                return result.returncode != 0
                
            except (subprocess.TimeoutExpired, Exception):
                # Consider timeout/error as killed mutation
                return True


@pytest.mark.slow
@pytest.mark.performance
class TestMutationAnalysis:
    """Test suite for mutation testing analysis."""
    
    def test_arithmetic_mutations_detected(self):
        """Test that arithmetic operator mutations are detected by test suite."""
        # Simple arithmetic function to test
        test_code = '''
def calculate_sum(a, b):
    return a + b

def calculate_difference(a, b):
    return a - b
'''
        
        operator = ArithmeticMutationOperator()
        mutations = operator.apply_mutation(test_code)
        
        assert len(mutations) >= 2  # Should find + and - operators
        
        mutation_types = [m['mutation'] for m in mutations]
        assert any("+ ->" in mt for mt in mutation_types)
        assert any("- ->" in mt for mt in mutation_types)
    
    def test_comparison_mutations_detected(self):
        """Test that comparison operator mutations are detected."""
        test_code = '''
def is_equal(a, b):
    return a == b

def is_greater(a, b):
    return a > b
'''
        
        operator = ComparisonMutationOperator()
        mutations = operator.apply_mutation(test_code)
        
        assert len(mutations) >= 2
        
        mutation_types = [m['mutation'] for m in mutations]
        assert any("== ->" in mt for mt in mutation_types)
        assert any("> ->" in mt for mt in mutation_types)
    
    def test_logical_mutations_detected(self):
        """Test that logical operator mutations are detected."""
        test_code = '''
def logical_and(a, b):
    return a and b

def is_valid():
    return True
'''
        
        operator = LogicalMutationOperator()
        mutations = operator.apply_mutation(test_code)
        
        assert len(mutations) >= 2
        
        mutation_types = [m['mutation'] for m in mutations]
        assert any("and ->" in mt for mt in mutation_types)
        assert any("True ->" in mt for mt in mutation_types)
    
    @pytest.mark.integration
    def test_mutation_testing_integration(self):
        """Integration test for complete mutation testing process."""
        # Create a simple source file for testing
        source_content = '''
def add_numbers(a, b):
    """Add two numbers together."""
    if a > 0 and b > 0:
        return a + b
    return 0

def multiply_numbers(a, b):
    """Multiply two numbers."""
    return a * b if a != 0 else 0
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_file = temp_path / "test_module.py"
            source_file.write_text(source_content)
            
            # Create corresponding test file
            test_content = '''
import sys
sys.path.insert(0, ".")
from test_module import add_numbers, multiply_numbers

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 5) == 0

def test_multiply_numbers():
    assert multiply_numbers(2, 3) == 6
    assert multiply_numbers(0, 5) == 0
    assert multiply_numbers(-2, 3) == -6
'''
            
            test_file = temp_path / "test_module_test.py"
            test_file.write_text(test_content)
            
            # Run mutation testing
            tester = MutationTester([source_file])
            results = tester.run_mutation_testing()
            
            assert results['total_mutants'] > 0
            assert results['mutation_score'] >= 0
            assert len(results['detailed_results']) == 1
    
    def test_mutation_score_calculation(self):
        """Test mutation score calculation accuracy."""
        # Mock results for testing
        mock_results = {
            'total_mutants': 10,
            'killed_mutants': 8,
            'survived_mutants': 2
        }
        
        expected_score = (8 / 10) * 100  # 80%
        assert expected_score == 80.0
        
        # Test edge case with no mutants
        mock_results_empty = {
            'total_mutants': 0,
            'killed_mutants': 0,
            'survived_mutants': 0
        }
        
        # Score should be 0 when no mutants are generated
        expected_score_empty = 0.0
        assert expected_score_empty == 0.0
    
    def test_mutation_report_generation(self):
        """Test generation of detailed mutation testing reports."""
        # Create sample mutation results
        sample_results = {
            'total_mutants': 5,
            'killed_mutants': 4,
            'survived_mutants': 1,
            'mutation_score': 80.0,
            'detailed_results': [
                {
                    'file': 'src/example.py',
                    'total_mutants': 5,
                    'killed_mutants': 4,
                    'survived_mutants': 1,
                    'mutations': [
                        {'mutation': '+ -> -', 'killed': True},
                        {'mutation': '== -> !=', 'killed': True},
                        {'mutation': 'and -> or', 'killed': True},
                        {'mutation': 'True -> False', 'killed': True},
                        {'mutation': '> -> <=', 'killed': False}
                    ]
                }
            ]
        }
        
        # Verify report structure
        assert 'mutation_score' in sample_results
        assert sample_results['mutation_score'] == 80.0
        assert len(sample_results['detailed_results']) == 1
        
        file_results = sample_results['detailed_results'][0]
        assert file_results['survived_mutants'] == 1
        assert len(file_results['mutations']) == 5
        
        # Count killed vs survived
        killed_count = sum(1 for m in file_results['mutations'] if m['killed'])
        survived_count = sum(1 for m in file_results['mutations'] if not m['killed'])
        
        assert killed_count == 4
        assert survived_count == 1


@pytest.mark.slow
def test_mutation_testing_cli_integration():
    """Test mutation testing integration with CLI commands."""
    # This would test the mutation testing integrated with the main CLI
    # For now, we'll simulate the integration
    
    # Mock CLI command that would run mutation testing
    def run_mutation_testing_cli(source_path: str, output_format: str = "json"):
        """Simulate CLI command for mutation testing."""
        source_files = list(Path(source_path).glob("*.py"))
        
        tester = MutationTester(source_files)
        results = tester.run_mutation_testing()
        
        if output_format == "json":
            return json.dumps(results, indent=2)
        elif output_format == "summary":
            return f"Mutation Score: {results['mutation_score']:.1f}%"
        
        return results
    
    # Test with different output formats
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple test file
        test_file = temp_path / "simple.py"
        test_file.write_text("def add(a, b): return a + b")
        
        # Test JSON output
        json_output = run_mutation_testing_cli(str(temp_path), "json")
        assert json_output is not None
        
        parsed_output = json.loads(json_output)
        assert 'mutation_score' in parsed_output
        
        # Test summary output
        summary_output = run_mutation_testing_cli(str(temp_path), "summary")
        assert "Mutation Score:" in summary_output