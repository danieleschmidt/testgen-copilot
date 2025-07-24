from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List

from .file_utils import safe_read_file, FileSizeError, safe_parse_ast
from .logging_config import get_security_logger
from .ast_utils import safe_parse_ast, ASTParsingError


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
        "tempfile.mktemp": "tempfile.mktemp is insecure; use NamedTemporaryFile",
    }

    def scan_file(self, path: str | Path) -> SecurityReport:
        logger = get_security_logger()
        file_path = Path(path)
        
        try:
            # Use safe_parse_ast for consolidated error handling
            try:
                result = safe_parse_ast(file_path, raise_on_syntax_error=False)
                if result is None:
                    # Syntax error occurred - safe_parse_ast already logged it
                    return SecurityReport(file_path, [SecurityIssue(0, "Syntax error in file")])
                tree, content = result
            except FileNotFoundError as e:
                logger.warning(f"File not found for security scan: {file_path}")
                return SecurityReport(file_path, [SecurityIssue(0, "File not found")])
            except PermissionError as e:
                logger.error(f"Cannot read file {file_path}: {e}")
                return SecurityReport(file_path, [SecurityIssue(0, f"Cannot read file: {e}")])
            except ValueError as e:  # UnicodeDecodeError wrapped in ValueError
                logger.warning(f"File encoding issue in {file_path}: {e}")
                return SecurityReport(file_path, [SecurityIssue(0, f"File encoding error: {e}")])
            except FileSizeError as e:
                logger.warning(f"File too large for security scan: {file_path}: {e}")
                return SecurityReport(file_path, [SecurityIssue(0, f"File too large: {e}")])
            except OSError as e:
                logger.error(f"File I/O error during security scan: {file_path}: {e}")
                return SecurityReport(file_path, [SecurityIssue(0, f"File I/O error: {e}")])
            
            try:
                tree = safe_parse_ast(content, file_path)
            except ASTParsingError as e:
                logger.warning("Skipping file due to parsing error", {
                    "file_path": str(file_path),
                    "error_message": str(e)
                })
                return SecurityReport(file_path, [SecurityIssue(e.line_number or 0, f"Syntax error: {e}")])
            
            issues: List[SecurityIssue] = []
            
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = self._full_name(node.func)
                        if not name:
                            continue
                        msg = self._dangerous_calls.get(name)
                        if msg:
                            if "subprocess" in name:
                                if not self._has_shell_true(node):
                                    continue
                            issues.append(SecurityIssue(node.lineno, msg))

                        if name == "os.system" or ("subprocess" in name and self._has_shell_true(node)):
                            if node.args and self._is_non_constant(node.args[0]):
                                issues.append(
                                    SecurityIssue(node.lineno, "Possible shell injection with dynamic command")
                                )

                        if name == "tempfile.NamedTemporaryFile":
                            for kw in node.keywords:
                                if kw.arg == "delete" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                                    issues.append(
                                        SecurityIssue(
                                            node.lineno,
                                            "NamedTemporaryFile(delete=False) is insecure; use delete=True",
                                        )
                                    )
            except Exception as e:
                logger.error(f"Error analyzing AST for {file_path}: {e}")
                issues.append(SecurityIssue(0, f"Analysis error: {e}"))
            
            logger.debug(f"Security scan of {file_path} found {len(issues)} issues")
            return SecurityReport(file_path, issues)
            
        except Exception as e:
            logger.error(f"Failed to scan file {file_path}: {e}")
            return SecurityReport(file_path, [SecurityIssue(0, f"Scan failed: {e}")])

    def scan_project(self, path: str | Path) -> List[SecurityReport]:
        logger = get_security_logger()
        base = Path(path)
        
        try:
            if not base.exists():
                logger.error(f"Project path does not exist: {base}")
                return [SecurityReport(base, [SecurityIssue(0, "Project path not found")])]
            
            if not base.is_dir():
                logger.error(f"Project path is not a directory: {base}")
                return [SecurityReport(base, [SecurityIssue(0, "Path is not a directory")])]
            
            reports = []
            python_files = list(base.rglob("*.py"))
            
            if not python_files:
                logger.warning(f"No Python files found in {base}")
                return [SecurityReport(base, [SecurityIssue(0, "No Python files found")])]
            
            logger.info(f"Scanning {len(python_files)} Python files for security issues")
            
            for file in python_files:
                try:
                    report = self.scan_file(file)
                    reports.append(report)
                except Exception as e:
                    logger.error(f"Failed to scan {file}: {e}")
                    reports.append(SecurityReport(file, [SecurityIssue(0, f"Scan error: {e}")]))
            
            total_issues = sum(len(r.issues) for r in reports)
            logger.info(f"Security scan completed: {total_issues} issues found across {len(reports)} files")
            return reports
            
        except Exception as e:
            logger.error(f"Failed to scan project {base}: {e}")
            return [SecurityReport(base, [SecurityIssue(0, f"Project scan failed: {e}")])]

    @staticmethod
    def _has_shell_true(node: ast.Call) -> bool:
        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
        return False

    @staticmethod
    def _is_non_constant(arg: ast.AST) -> bool:
        return not isinstance(arg, ast.Constant)

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
