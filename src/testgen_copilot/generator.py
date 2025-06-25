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

        lines: List[str] = ["\n".join(imports), ""]

        for func in functions:
            lines.extend(self._build_test_for_function(source_path, func))
            lines.append("")

            if self.config.include_edge_cases:
                lines.extend(self._build_edge_case_test(source_path, func))
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

        body.append("    # TODO: assert expected result")
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

        lines.append("    # TODO: assert edge case result")
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
            lines.append("  // TODO: expect result")
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
            lines.append(f"        // TODO: call {m} and assert")
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
            lines.append(f"        // TODO: call {m} and Assert")
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
            lines.append("    _ = result // TODO: use result")
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
            lines.append("        // TODO: assert result")
            lines.append("    }\n")
        lines.append("}")

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{source_path.stem}_test.rs"
        out_file.write_text("\n".join(lines).rstrip() + "\n")
        return out_file

    def _parse_rust_functions(self, path: Path) -> List[str]:
        text = path.read_text()
        return re.findall(r"fn\s+(\w+)\s*\(", text) or ["func_under_test"]
