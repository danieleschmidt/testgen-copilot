"""TestGen Copilot package."""

from .core import identity
from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner, SecurityIssue, SecurityReport
from .vscode import scaffold_extension, suggest_from_diagnostics, write_usage_docs

__all__ = [
    "identity",
    "GenerationConfig",
    "TestGenerator",
    "SecurityScanner",
    "SecurityIssue",
    "SecurityReport",
    "scaffold_extension",
    "suggest_from_diagnostics",
    "write_usage_docs",
]
