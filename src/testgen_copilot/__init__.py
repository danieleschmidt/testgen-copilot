"""TestGen Copilot package."""

from .core import identity
from .generator import GenerationConfig, TestGenerator
from .security import SecurityScanner, SecurityIssue, SecurityReport
from .security_rules import SecurityRulesManager, SecurityRule, SecurityRuleSet
from .version import get_package_version, get_version_info, __version__
from .vscode import scaffold_extension, suggest_from_diagnostics, write_usage_docs
from .coverage import CoverageAnalyzer, ParallelCoverageAnalyzer, CoverageResult, StreamingCoverageAnalyzer
from .streaming import StreamingProcessor, FileStreamProcessor, StreamingProgress, create_progress_reporter
from .quality import TestQualityScorer
from .file_utils import safe_read_file, FileSizeError, safe_parse_ast, SyntaxErrorStrategy
from .resource_limits import (
    MemoryMonitor, BatchProcessor, TimeoutHandler, CrossPlatformTimeoutHandler, ResourceMemoryError,
    ResourceLimits, ResourceMonitor, safe_parse_ast_with_timeout, validate_test_content,
    AST_PARSE_TIMEOUT, MAX_PROJECT_FILES
)
from .cache import (
    LRUCache, CacheEntry, cached_operation, get_cache_stats, clear_all_caches,
    ast_cache, file_content_cache, analysis_cache
)
from .autonomous_backlog import BacklogManager, BacklogItem, TaskType, TaskStatus, RiskTier
from .autonomous_execution import TDDExecutor, SecurityChecker, ExecutionResult, ExecutionPhase
from .autonomous_manager import AutonomousManager
from .metrics_collector import MetricsCollector, DORAMetrics, CIMetrics, OperationalMetrics
from .database import (
    DatabaseConnection, get_database, AnalysisResult, TestCase, SecurityIssue,
    ProjectMetrics, ProcessingSession, AnalysisRepository, TestCaseRepository,
    SecurityRepository, MetricsRepository, SessionRepository, run_migrations
)

__all__ = [
    "identity",
    "GenerationConfig",
    "TestGenerator",
    "SecurityScanner",
    "SecurityIssue",
    "SecurityReport",
    "SecurityRulesManager",
    "SecurityRule",
    "SecurityRuleSet",
    "scaffold_extension",
    "suggest_from_diagnostics",
    "write_usage_docs",
    "CoverageAnalyzer",
    "ParallelCoverageAnalyzer",
    "CoverageResult",
    "StreamingCoverageAnalyzer",
    "StreamingProcessor",
    "FileStreamProcessor",
    "StreamingProgress",
    "create_progress_reporter",
    "TestQualityScorer",
    "safe_read_file",
    "FileSizeError",
    "safe_parse_ast",
    "SyntaxErrorStrategy",
    "MemoryMonitor",
    "BatchProcessor",
    "TimeoutHandler",
    "CrossPlatformTimeoutHandler",
    "ResourceMemoryError",
    "safe_parse_ast_with_timeout",
    "validate_test_content",
    "AST_PARSE_TIMEOUT",
    "MAX_PROJECT_FILES",
    "ResourceLimits",
    "ResourceMonitor", 
    "CircuitBreaker",
    "safe_parse_ast",
    "ASTParsingError",
    "extract_functions",
    "extract_classes",
    "LRUCache",
    "CacheEntry", 
    "cached_operation",
    "get_cache_stats",
    "clear_all_caches",
    "ast_cache",
    "file_content_cache",
    "analysis_cache",
    "get_package_version",
    "get_version_info",
    "__version__",
    "BacklogManager",
    "BacklogItem", 
    "TaskType",
    "TaskStatus",
    "RiskTier",
    "TDDExecutor",
    "SecurityChecker",
    "ExecutionResult",
    "ExecutionPhase",
    "AutonomousManager",
    "MetricsCollector",
    "DORAMetrics",
    "CIMetrics",
    "OperationalMetrics",
    "DatabaseConnection",
    "get_database",
    "AnalysisResult",
    "TestCase",
    "SecurityIssue",
    "ProjectMetrics",
    "ProcessingSession",
    "AnalysisRepository",
    "TestCaseRepository",
    "SecurityRepository",
    "MetricsRepository",
    "SessionRepository",
    "run_migrations"
]
