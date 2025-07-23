"""TestGen Copilot package."""

from .core import identity
from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner, SecurityIssue, SecurityReport
from .vscode import scaffold_extension, suggest_from_diagnostics, write_usage_docs
from .coverage import CoverageAnalyzer, ParallelCoverageAnalyzer, CoverageResult
from .quality import TestQualityScorer
from .file_utils import safe_read_file, FileSizeError

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
    "CoverageAnalyzer",
    "ParallelCoverageAnalyzer",
    "CoverageResult",
    "TestQualityScorer",
    "safe_read_file",
    "FileSizeError",
]
