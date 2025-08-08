"""Data models for TestGen Copilot database persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SecuritySeverity(Enum):
    """Security issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessingSession:
    """Represents a processing session for tracking work."""
    id: Optional[int] = None
    session_id: str = ""
    project_path: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project_path": self.project_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "configuration": self.configuration
        }


@dataclass
class AnalysisResult:
    """Represents the result of analyzing a source file."""
    id: Optional[int] = None
    session_id: str = ""
    file_path: str = ""
    language: str = ""
    processing_time_ms: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    tests_generated: Optional[str] = None
    coverage_percentage: Optional[float] = None
    quality_score: Optional[float] = None
    security_issues_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "file_path": self.file_path,
            "language": self.language,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status.value,
            "tests_generated": self.tests_generated,
            "coverage_percentage": self.coverage_percentage,
            "quality_score": self.quality_score,
            "security_issues_count": self.security_issues_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TestCase:
    """Represents a generated test case."""
    id: Optional[int] = None
    analysis_result_id: int = 0
    test_name: str = ""
    test_type: str = "unit"  # unit, integration, edge_case, error_path, benchmark
    function_name: str = ""
    test_content: str = ""
    assertions_count: int = 0
    complexity_score: float = 0.0
    execution_time_ms: Optional[int] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "analysis_result_id": self.analysis_result_id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "function_name": self.function_name,
            "test_content": self.test_content,
            "assertions_count": self.assertions_count,
            "complexity_score": self.complexity_score,
            "execution_time_ms": self.execution_time_ms,
            "passed": self.passed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SecurityIssue:
    """Represents a security issue found during analysis."""
    id: Optional[int] = None
    analysis_result_id: int = 0
    rule_id: str = ""
    rule_name: str = ""
    severity: SecuritySeverity = SecuritySeverity.LOW
    category: str = ""
    description: str = ""
    file_path: str = ""
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: str = ""
    recommendation: str = ""
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    confidence: float = 0.0
    false_positive: bool = False
    suppressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "analysis_result_id": self.analysis_result_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "confidence": self.confidence,
            "false_positive": self.false_positive,
            "suppressed": self.suppressed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ProjectMetrics:
    """Represents aggregated metrics for a project or session."""
    id: Optional[int] = None
    session_id: str = ""
    project_path: str = ""
    total_files: int = 0
    analyzed_files: int = 0
    generated_tests: int = 0
    total_test_cases: int = 0
    average_coverage: float = 0.0
    average_quality_score: float = 0.0
    security_issues_total: int = 0
    security_issues_critical: int = 0
    security_issues_high: int = 0
    security_issues_medium: int = 0
    security_issues_low: int = 0
    processing_time_total_ms: int = 0
    processing_time_average_ms: float = 0.0
    lines_of_code: int = 0
    lines_of_tests: int = 0
    test_to_code_ratio: float = 0.0
    languages_used: List[str] = field(default_factory=list)
    frameworks_detected: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project_path": self.project_path,
            "total_files": self.total_files,
            "analyzed_files": self.analyzed_files,
            "generated_tests": self.generated_tests,
            "total_test_cases": self.total_test_cases,
            "average_coverage": self.average_coverage,
            "average_quality_score": self.average_quality_score,
            "security_issues_total": self.security_issues_total,
            "security_issues_critical": self.security_issues_critical,
            "security_issues_high": self.security_issues_high,
            "security_issues_medium": self.security_issues_medium,
            "security_issues_low": self.security_issues_low,
            "processing_time_total_ms": self.processing_time_total_ms,
            "processing_time_average_ms": self.processing_time_average_ms,
            "lines_of_code": self.lines_of_code,
            "lines_of_tests": self.lines_of_tests,
            "test_to_code_ratio": self.test_to_code_ratio,
            "languages_used": self.languages_used,
            "frameworks_detected": self.frameworks_detected,
            "metadata": self.metadata,
            "calculated_at": self.calculated_at.isoformat()
        }
