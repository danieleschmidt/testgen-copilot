# Mutation Testing Configuration
# Pytest configuration for mutation testing scenarios

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def mutation_test_env():
    """Create isolated environment for mutation testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy source files to temporary directory
        src_path = Path("src/testgen_copilot")
        temp_src = temp_path / "testgen_copilot"
        temp_src.mkdir(parents=True, exist_ok=True)
        
        # Copy essential modules for testing
        for py_file in src_path.glob("*.py"):
            if py_file.name not in ["__pycache__"]:
                shutil.copy2(py_file, temp_src)
        
        yield {
            "temp_path": temp_path,
            "src_path": temp_src,
            "original_src": src_path
        }


@pytest.fixture
def mutation_scenarios():
    """Define common mutation scenarios for testing."""
    return [
        {
            "name": "arithmetic_operators",
            "mutations": [
                ("+", "-"),
                ("-", "+"),
                ("*", "/"),
                ("/", "*"),
                ("//", "%"),
                ("%", "//")
            ]
        },
        {
            "name": "comparison_operators",
            "mutations": [
                ("==", "!="),
                ("!=", "=="),
                ("<", ">="),
                (">=", "<"),
                (">", "<="),
                ("<=", ">")
            ]
        },
        {
            "name": "logical_operators",
            "mutations": [
                ("and", "or"),
                ("or", "and"),
                ("not", ""),
                ("True", "False"),
                ("False", "True")
            ]
        },
        {
            "name": "boundary_values",
            "mutations": [
                ("0", "1"),
                ("1", "0"),
                ("[]", "[None]"),
                ("{}", "{'_': None}"),
                ("''", "'_'"),
                ('""', '"_"')
            ]
        }
    ]