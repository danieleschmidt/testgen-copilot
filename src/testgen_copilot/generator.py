"""Utilities for generating test stubs from source files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List


@dataclass
class GenerationConfig:
    """Configuration for :class:`TestGenerator`."""

    language: str = "python"
    include_edge_cases: bool = True
    include_error_paths: bool = True
    include_benchmarks: bool = True
    include_integration_tests: bool = True
    use_mocking: bool = True


class TestGenerator:
    """Create simple unit test stubs for multiple languages."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_tests(self, file_path: str | Path, output_dir: str | Path) -> Path:
        """Generate tests for ``file_path`` inside ``output_dir``."""

        lang = self.config.language.lower()
        if lang in {"python", "py"}:
            return self._generate_python_tests(Path(file_path), Path(output_dir))
        if lang in {"javascript", "js", "typescript", "ts"}:
            return self._generate_javascript_tests(Path(file_path), Path(output_dir))
        if lang == "java":
            return self._generate_java_tests(Path(file_path), Path(output_dir))
        if lang in {"c#", "csharp"}:
            return self._generate_csharp_tests(Path(file_path), Path(output_dir))
        if lang == "go":
            return self._generate_go_tests(Path(file_path), Path(output_dir))
        if lang == "rust":
            return self._generate_rust_tests(Path(file_path), Path(output_dir))
        raise ValueError(f"Unsupported language: {self.config.language}")

    # ------------------------------------------------------------------
    # Python generation
    # ------------------------------------------------------------------
    def _generate_python_tests(self, source_path: Path, out_dir: Path) -> Path:
        functions = self._parse_functions(source_path)
        test_content = self._build_test_file(source_path, functions)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"test_{source_path.stem}.py"
        out_file.write_text(test_content)
        return out_file

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    def _parse_functions(self, path: Path) -> List[ast.FunctionDef]:
        tree = ast.parse(path.read_text())
        return [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    def _build_test_file(
        self, source_path: Path, functions: Iterable[ast.FunctionDef]
    ) -> str:
        imports = [f"import {source_path.stem}"]

        if any(self._uses_open(func) for func in functions) and self.config.use_mocking:
            imports.append("from unittest.mock import mock_open, patch")
        if (
            self.config.include_error_paths
            and any(self._exception_names(f) for f in functions)
        ) or self.config.include_benchmarks:
            imports.append("import pytest")

        lines: List[str] = ["\n".join(imports), ""]

        for func in functions:
            lines.extend(self._build_test_for_function(source_path, func))
            lines.append("")

            if self.config.include_edge_cases:
                lines.extend(self._build_edge_case_test(source_path, func))
                lines.append("")

            if self.config.include_error_paths:
                for exc in self._exception_names(func):
                    lines.extend(self._build_error_test(source_path, func, exc))
                    lines.append("")

            if self.config.include_benchmarks:
                lines.extend(self._build_benchmark_test(source_path, func))
                lines.append("")

        if self.config.include_integration_tests and len(functions) > 1:
            lines.extend(self._build_integration_test(source_path, functions))
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _build_test_for_function(
        self, source_path: Path, func: ast.FunctionDef
    ) -> List[str]:
        """Build the main test function body."""

        args = [arg.arg for arg in func.args.args]
        call_args = ", ".join(self._default_value(arg) for arg in args)

        body: List[str] = [f"def test_{func.name}():"]

        if self._uses_open(func) and self.config.use_mocking:
            body.append("    with patch('builtins.open', mock_open(read_data='data')):")
            body.append(f"        result = {source_path.stem}.{func.name}({call_args})")
        else:
            body.append(f"    result = {source_path.stem}.{func.name}({call_args})")

        # Generate appropriate assertion based on function analysis
        assertion = self._generate_assertion(func, "result")
        body.append(f"    {assertion}")
        return body

    def _build_edge_case_test(
        self, source_path: Path, func: ast.FunctionDef
    ) -> List[str]:
        """Build an additional edge case test."""

        args = [arg.arg for arg in func.args.args]
        call_args = ", ".join(self._edge_case_value(arg) for arg in args)

        lines: List[str] = [f"def test_{func.name}_edge_case():"]

        if self._uses_open(func) and self.config.use_mocking:
            lines.append(
                "    with patch('builtins.open', mock_open(read_data='data')):"
            )
            lines.append(
                f"        result = {source_path.stem}.{func.name}({call_args})"
            )
        else:
            lines.append(f"    result = {source_path.stem}.{func.name}({call_args})")

        # Generate appropriate assertion for edge case
        assertion = self._generate_edge_case_assertion(func, "result")
        lines.append(f"    {assertion}")
        return lines

    @staticmethod
    def _uses_open(func: ast.FunctionDef) -> bool:
        return any(
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open"
            for node in ast.walk(func)
        )

    @staticmethod
    def _default_value(name: str) -> str:
        name_l = name.lower()
        if "path" in name_l or "file" in name_l:
            return "tmp_path/'sample.txt'"
        return "1"

    @staticmethod
    def _edge_case_value(name: str) -> str:
        name_l = name.lower()
        if "path" in name_l or "file" in name_l:
            return "''"
        return "0"

    @staticmethod
    def _exception_names(func: ast.FunctionDef) -> List[str]:
        """Return list of exception names raised in ``func``."""
        names = []
        for node in ast.walk(func):
            if isinstance(node, ast.Raise):
                exc = node.exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name):
                    names.append(exc.id)
                elif exc is not None:
                    names.append("Exception")
        return names

    def _build_error_test(
        self, source_path: Path, func: ast.FunctionDef, exc: str
    ) -> List[str]:
        args = [arg.arg for arg in func.args.args]
        call_args = ", ".join(self._default_value(arg) for arg in args)

        lines = [f"def test_{func.name}_raises_{exc.lower()}():"]
        lines.append(f"    with pytest.raises({exc}):")
        lines.append(f"        {source_path.stem}.{func.name}({call_args})")
        return lines

    def _build_benchmark_test(
        self, source_path: Path, func: ast.FunctionDef
    ) -> List[str]:
        args = [arg.arg for arg in func.args.args]
        call_args = ", ".join(self._default_value(arg) for arg in args)

        lines = [f"def test_{func.name}_benchmark(benchmark):"]
        if self._uses_open(func) and self.config.use_mocking:
            lines.append("    with patch('builtins.open', mock_open(read_data='data')):")
            lines.append(
                f"        benchmark({source_path.stem}.{func.name}, {call_args})"
            )
        else:
            lines.append(
                f"    benchmark({source_path.stem}.{func.name}, {call_args})"
            )
        return lines

    def _build_integration_test(
        self, source_path: Path, functions: Iterable[ast.FunctionDef]
    ) -> List[str]:
        """Build a simple integration test calling all functions."""
        lines = [f"def test_{source_path.stem}_integration():"]
        for func in functions:
            args = [arg.arg for arg in func.args.args]
            call_args = ", ".join(self._default_value(arg) for arg in args)
            if self._uses_open(func) and self.config.use_mocking:
                lines.append(
                    "    with patch('builtins.open', mock_open(read_data='data')):"  # noqa: E501
                )
                lines.append(
                    f"        {source_path.stem}.{func.name}({call_args})"
                )
            else:
                lines.append(
                    f"    {source_path.stem}.{func.name}({call_args})"
                )
        return lines

    # ------------------------------------------------------------------
    # JavaScript / TypeScript generation
    # ------------------------------------------------------------------
    def _generate_javascript_tests(self, source_path: Path, out_dir: Path) -> Path:
        names = self._parse_js_functions(source_path)
        lines: List[str] = [
            f"import {{ {', '.join(names)} }} from './{source_path.stem}';",
            "",
        ]
        for name in names:
            lines.append(f"test('{name} works', () => {{")
            lines.append(f"  const result = {name}();")
            lines.append("  expect(result).toBeDefined();")
            lines.append("});\n")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{source_path.stem}.test.js"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_js_functions(self, path: Path) -> List[str]:
        text = path.read_text()
        patterns = [r"function\s+(\w+)\s*\(", r"const\s+(\w+)\s*=\s*\("]
        names = []
        for pat in patterns:
            names.extend(re.findall(pat, text))
        return list(dict.fromkeys(names)) or [path.stem]

    # ------------------------------------------------------------------
    # Java generation
    # ------------------------------------------------------------------
    def _generate_java_tests(self, source_path: Path, out_dir: Path) -> Path:
        methods = self._parse_java_methods(source_path)
        class_name = source_path.stem.capitalize() + "Test"
        lines: List[str] = [
            "import org.junit.jupiter.api.Test;",
            f"class {class_name} {{",
            "",
        ]
        for m in methods:
            lines.append("    @Test")
            lines.append(f"    void {m}() {{")
            lines.append(f"        {m}();")
            lines.append("        // Verify method executes without throwing")
            lines.append("    }\n")
        lines.append("}")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{class_name}.java"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_java_methods(self, path: Path) -> List[str]:
        text = path.read_text()
        return re.findall(r"public\s+\w+\s+(\w+)\s*\(", text) or ["methodUnderTest"]

    # ------------------------------------------------------------------
    # C# generation
    # ------------------------------------------------------------------
    def _generate_csharp_tests(self, source_path: Path, out_dir: Path) -> Path:
        methods = self._parse_csharp_methods(source_path)
        class_name = source_path.stem + "Tests"
        lines: List[str] = [
            "using NUnit.Framework;",
            f"public class {class_name} {{",
            "",
        ]
        for m in methods:
            lines.append("    [Test]")
            lines.append(f"    public void {m}() {{")
            lines.append(f"        {m}();")
            lines.append("        // Verify method executes without throwing")
            lines.append("    }\n")
        lines.append("}")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{class_name}.cs"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_csharp_methods(self, path: Path) -> List[str]:
        text = path.read_text()
        return re.findall(r"public\s+\w+\s+(\w+)\s*\(", text) or ["MethodUnderTest"]

    # ------------------------------------------------------------------
    # Go generation
    # ------------------------------------------------------------------
    def _generate_go_tests(self, source_path: Path, out_dir: Path) -> Path:
        funcs = self._parse_go_functions(source_path)
        lines: List[str] = ["package main", "", 'import "testing"', ""]
        for f_name in funcs:
            lines.append(f"func Test{f_name.capitalize()}(t *testing.T) {{")
            lines.append(f"    result := {f_name}()")
            lines.append("    if result != nil {")
            lines.append("        t.Errorf(\"Expected nil, got %v\", result)")
            lines.append("    }")
            lines.append("}\n")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{source_path.stem}_test.go"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_go_functions(self, path: Path) -> List[str]:
        text = path.read_text()
        return re.findall(r"func\s+(\w+)\s*\(", text) or ["FuncUnderTest"]

    # ------------------------------------------------------------------
    # Rust generation
    # ------------------------------------------------------------------
    def _generate_rust_tests(self, source_path: Path, out_dir: Path) -> Path:
        funcs = self._parse_rust_functions(source_path)
        lines: List[str] = ["#[cfg(test)]", "mod tests {", "    use super::*;", ""]
        for f_name in funcs:
            lines.append("    #[test]")
            lines.append(f"    fn {f_name}_test() {{")
            lines.append(f"        let result = {f_name}();")
            lines.append("        assert!(result.is_ok());")
            lines.append("    }\n")
        lines.append("}")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{source_path.stem}_test.rs"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_rust_functions(self, path: Path) -> List[str]:
        text = path.read_text()
        return re.findall(r"fn\s+(\w+)\s*\(", text) or ["func_under_test"]

    # ------------------------------------------------------------------
    # Assertion generation helpers
    # ------------------------------------------------------------------
    def _generate_assertion(self, func: ast.FunctionDef, result_var: str) -> str:
        """Generate appropriate assertion based on function analysis."""
        # Analyze function to determine likely return type and assertion
        return_hints = self._analyze_return_type(func)
        
        if "bool" in return_hints.lower():
            return f"assert isinstance({result_var}, bool)"
        elif "str" in return_hints.lower() or "string" in return_hints.lower():
            return f"assert isinstance({result_var}, str)"
        elif "int" in return_hints.lower() or "number" in return_hints.lower():
            return f"assert isinstance({result_var}, (int, float))"
        elif "list" in return_hints.lower() or "array" in return_hints.lower():
            return f"assert isinstance({result_var}, list)"
        elif "dict" in return_hints.lower() or "object" in return_hints.lower():
            return f"assert isinstance({result_var}, dict)"
        elif self._has_return_statement(func):
            return f"assert {result_var} is not None"
        else:
            return f"assert {result_var} is None"

    def _generate_edge_case_assertion(self, func: ast.FunctionDef, result_var: str) -> str:
        """Generate assertion for edge case scenarios."""
        # For edge cases, often check for specific boundary conditions
        if self._raises_exceptions(func):
            return f"# Edge case may raise exception or return special value"
        
        return_hints = self._analyze_return_type(func)
        if "bool" in return_hints.lower():
            return f"assert isinstance({result_var}, bool)"
        elif "str" in return_hints.lower():
            return f"assert {result_var} == '' or isinstance({result_var}, str)"
        elif "int" in return_hints.lower() or "number" in return_hints.lower():
            return f"assert {result_var} == 0 or isinstance({result_var}, (int, float))"
        else:
            return f"assert {result_var} is not None or {result_var} is None"

    def _analyze_return_type(self, func: ast.FunctionDef) -> str:
        """Analyze function to determine likely return type."""
        # Check for type annotations first
        if func.returns:
            if isinstance(func.returns, ast.Name):
                return func.returns.id
            elif isinstance(func.returns, ast.Constant):
                return str(func.returns.value)
        
        # Analyze docstring for type hints
        docstring = ast.get_docstring(func)
        if docstring:
            doc_lower = docstring.lower()
            if "return" in doc_lower:
                if "bool" in doc_lower or "true" in doc_lower or "false" in doc_lower:
                    return "bool"
                elif "str" in doc_lower or "string" in doc_lower:
                    return "str"
                elif "int" in doc_lower or "number" in doc_lower or "float" in doc_lower:
                    return "int"
                elif "list" in doc_lower or "array" in doc_lower:
                    return "list"
                elif "dict" in doc_lower or "dictionary" in doc_lower:
                    return "dict"
        
        # Analyze return statements
        return_types = []
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Constant):
                    return_types.append(type(node.value.value).__name__)
                elif isinstance(node.value, ast.BinOp):
                    return_types.append("int")  # Assume arithmetic operations return numbers
                elif isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in ["str", "format"]:
                            return_types.append("str")
                        elif node.value.func.id in ["int", "float"]:
                            return_types.append("int")
                        elif node.value.func.id in ["list", "[]"]:
                            return_types.append("list")
                        elif node.value.func.id in ["dict", "{}"]:
                            return_types.append("dict")
        
        return return_types[0] if return_types else "unknown"

    def _has_return_statement(self, func: ast.FunctionDef) -> bool:
        """Check if function has explicit return statements."""
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False

    def _raises_exceptions(self, func: ast.FunctionDef) -> bool:
        """Check if function explicitly raises exceptions."""
        return bool(self._exception_names(func))
