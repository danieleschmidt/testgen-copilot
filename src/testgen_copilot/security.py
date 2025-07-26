from __future__ import annotations

import ast
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List

from .file_utils import safe_read_file, FileSizeError, safe_parse_ast
from .logging_config import get_security_logger
from .ast_utils import ASTParsingError
from .security_rules import SecurityRulesManager
from .cache import cached_operation, analysis_cache


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

    def __init__(self, config_path: str = None):
        """Initialize scanner with optional custom configuration path."""
        self.rules_manager = SecurityRulesManager(config_path)
        self._dangerous_calls = None
        self._shell_required_patterns = None
        self._dynamic_check_patterns = None
    
    @property
    def dangerous_calls(self) -> dict:
        """Get dangerous calls dictionary, loading rules if needed."""
        if self._dangerous_calls is None:
            self._dangerous_calls = self.rules_manager.get_dangerous_calls_dict()
        return self._dangerous_calls
    
    @property
    def shell_required_patterns(self) -> list:
        """Get patterns that require shell=True checking."""
        if self._shell_required_patterns is None:
            self._shell_required_patterns = self.rules_manager.get_rules_requiring_shell()
        return self._shell_required_patterns
    
    @property
    def dynamic_check_patterns(self) -> list:
        """Get patterns that need dynamic argument checking."""
        if self._dynamic_check_patterns is None:
            self._dynamic_check_patterns = self.rules_manager.get_rules_checking_dynamic_args()
        return self._dynamic_check_patterns

    @cached_operation("security_scan_file", analysis_cache)
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
            
            issues: List[SecurityIssue] = []
            
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = self._full_name(node.func)
                        if not name:
                            continue
                        msg = self.dangerous_calls.get(name)
                        if msg:
                            # Check if this pattern requires shell=True
                            if name in self.shell_required_patterns:
                                if not self._has_shell_true(node):
                                    continue
                            issues.append(SecurityIssue(node.lineno, msg))

                        # Check for dynamic arguments in patterns that need it
                        if name in self.dynamic_check_patterns:
                            # For subprocess calls, also check if shell=True
                            if "subprocess" in name and not self._has_shell_true(node):
                                continue
                            
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
