from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SecurityIssue:
    """Representation of a potential security problem."""

    line: int
    message: str


@dataclass
class SecurityReport:
    """Report produced after scanning a file."""

    path: Path
    issues: List[SecurityIssue]

    def to_text(self) -> str:
        if not self.issues:
            return f"{self.path}: no issues found"
        lines = [f"{self.path}:"]
        for issue in self.issues:
            lines.append(f"  Line {issue.line}: {issue.message}")
        return "\n".join(lines)


class SecurityScanner:
    """Simple static analyzer for insecure code patterns."""

    _dangerous_calls = {
        "eval": "Use of eval() can lead to code execution vulnerabilities",
        "exec": "Use of exec() can lead to code execution vulnerabilities",
        "os.system": "os.system() can be unsafe; prefer subprocess without shell=True",
        "subprocess.call": "subprocess.call with shell=True can be unsafe",
        "subprocess.Popen": "subprocess.Popen with shell=True can be unsafe",
        "subprocess.run": "subprocess.run with shell=True can be unsafe",
        "pickle.load": "Deserializing with pickle can be unsafe with untrusted data",
        "pickle.loads": "Deserializing with pickle can be unsafe with untrusted data",
        "yaml.load": "yaml.load is unsafe; use yaml.safe_load instead",
    }

    def scan_file(self, path: str | Path) -> SecurityReport:
        file_path = Path(path)
        try:
            tree = ast.parse(file_path.read_text())
        except (OSError, SyntaxError):
            return SecurityReport(file_path, [])

        issues: List[SecurityIssue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._full_name(node.func)
                if not name:
                    continue
                msg = self._dangerous_calls.get(name)
                if msg:
                    if "subprocess" in name or name == "os.system":
                        if not self._has_shell_true(node):
                            continue
                    issues.append(SecurityIssue(node.lineno, msg))
        return SecurityReport(file_path, issues)

    def scan_project(self, path: str | Path) -> List[SecurityReport]:
        base = Path(path)
        reports = []
        for file in base.rglob("*.py"):
            reports.append(self.scan_file(file))
        return reports

    @staticmethod
    def _has_shell_true(node: ast.Call) -> bool:
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
        return False

    @staticmethod
    def _full_name(func: ast.AST) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
                return ".".join(reversed(parts))
        return None
