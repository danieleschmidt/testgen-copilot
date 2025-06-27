"""Simple code coverage estimation utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


class CoverageAnalyzer:
    """Estimate coverage of tests over a source file."""

    def analyze(self, source_path: str | Path, tests_dir: str | Path) -> float:
        """Return the percentage of source functions referenced in tests."""
        src = Path(source_path)
        tests = Path(tests_dir)

        func_names = self._functions_in_file(src)
        if not func_names:
            return 100.0

        covered = self._functions_used_in_tests(tests, func_names)
        return (len(covered) / len(func_names)) * 100

    def uncovered_functions(
        self, source_path: str | Path, tests_dir: str | Path
    ) -> set[str]:
        """Return names of functions in ``source_path`` not referenced by tests."""
        src = Path(source_path)
        tests = Path(tests_dir)

        func_names = self._functions_in_file(src)
        if not func_names:
            return set()

        covered = self._functions_used_in_tests(tests, func_names)
        return set(func_names) - covered

    @staticmethod
    def _functions_in_file(path: Path) -> list[str]:
        """Return all function and method names defined in ``path``."""
        tree = ast.parse(path.read_text())
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    @staticmethod
    def _functions_used_in_tests(tests_dir: Path, func_names: Iterable[str]) -> set[str]:
        """Return the subset of ``func_names`` referenced within ``tests_dir``."""
        names = set(func_names)
        covered: set[str] = set()
        for test_file in tests_dir.rglob("test_*.py"):
            tree = ast.parse(test_file.read_text())

            # Track aliases from ``from x import y as z`` so calls to ``z`` are
            # associated with ``y`` when checking coverage.
            alias_map = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.asname and alias.name in names:
                            alias_map[alias.asname] = alias.name

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        target = alias_map.get(node.id, node.id)
                        if target in names:
                            covered.add(target)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.ctx, ast.Load) and node.attr in names:
                        covered.add(node.attr)
        return covered
