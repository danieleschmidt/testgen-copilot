"""Utilities to score the quality of test files."""

from __future__ import annotations

import ast
from pathlib import Path


class TestQualityScorer:
    """Estimate quality of tests based on presence of assertions."""

    def score(self, tests_dir: str | Path) -> float:
        """Return percentage of test functions containing ``assert`` statements."""
        tests = Path(tests_dir)
        total = 0
        with_assert = 0
        for path in tests.rglob("test_*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    total += 1
                    if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                        with_assert += 1
        if total == 0:
            return 100.0
        return (with_assert / total) * 100

    def low_quality_tests(self, tests_dir: str | Path) -> set[str]:
        """Return names of test functions lacking assertions."""
        tests = Path(tests_dir)
        lacking: set[str] = set()
        for path in tests.rglob("test_*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                    if not any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                        lacking.add(node.name)
        return lacking
