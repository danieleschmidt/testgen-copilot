"""Pydantic models for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Status of processing operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SecuritySeverity(str, Enum):
    """Security issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisPhase(str, Enum):
    """Phases of code analysis."""
    PARSING = "parsing"
    SECURITY_SCAN = "security_scan"
    TEST_GENERATION = "test_generation"
    COVERAGE_ANALYSIS = "coverage_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"


# Request Models
class AnalysisRequest(BaseModel):
    """Request model for single file analysis."""
    file_path: str = Field(..., description="Path to the source file to analyze")
    output_dir: str = Field(..., description="Directory to save generated tests")
    language: Optional[str] = Field(None, description="Programming language (auto-detected if not provided)")

    # Analysis configuration
    include_edge_cases: bool = Field(True, description="Generate edge case tests")
    include_error_paths: bool = Field(True, description="Generate error handling tests")
    include_benchmarks: bool = Field(False, description="Generate benchmark tests")
    include_integration_tests: bool = Field(True, description="Generate integration tests")
    use_mocking: bool = Field(True, description="Use mocking in generated tests")

    # Analysis phases to run
    phases: Optional[List[AnalysisPhase]] = Field(
        None,
        description="Specific analysis phases to run (all phases if not specified)"
    )

    # Security scanning options
    enable_security_scan: bool = Field(True, description="Enable security vulnerability scanning")
    security_rules_strict: bool = Field(False, description="Use strict security rules")

    # Coverage and quality options
    enable_coverage_analysis: bool = Field(True, description="Enable code coverage analysis")
    enable_quality_assessment: bool = Field(True, description="Enable test quality assessment")
    coverage_target: Optional[float] = Field(85.0, ge=0, le=100, description="Target coverage percentage")
    quality_target: Optional[float] = Field(75.0, ge=0, le=100, description="Target quality score")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate that file path exists and is a file."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return str(path.absolute())

    @validator('output_dir')
    def validate_output_dir(cls, v):
        """Validate output directory path."""
        path = Path(v)
        return str(path.absolute())


class ProjectAnalysisRequest(BaseModel):
    """Request model for project-wide analysis."""
    project_path: str = Field(..., description="Path to the project root directory")
    output_dir: str = Field(..., description="Directory to save generated tests")

    # File filtering
    file_patterns: Optional[List[str]] = Field(
        ["*.py", "*.js", "*.ts", "*.java", "*.cs", "*.go", "*.rs"],
        description="File patterns to include in analysis"
    )
    exclude_patterns: Optional[List[str]] = Field(
        ["*test*", "*spec*", "__pycache__", "node_modules", ".git"],
        description="Patterns to exclude from analysis"
    )

    # Analysis configuration (inherited from AnalysisRequest)
    include_edge_cases: bool = Field(True, description="Generate edge case tests")
    include_error_paths: bool = Field(True, description="Generate error handling tests")
    include_benchmarks: bool = Field(False, description="Generate benchmark tests")
    include_integration_tests: bool = Field(True, description="Generate integration tests")
    use_mocking: bool = Field(True, description="Use mocking in generated tests")

    # Processing options
    concurrent_limit: int = Field(4, ge=1, le=16, description="Maximum concurrent file processing")
    batch_size: Optional[int] = Field(None, ge=1, description="Batch size for processing (optional)")

    # Analysis phases and options
    phases: Optional[List[AnalysisPhase]] = Field(None, description="Analysis phases to run")
    enable_security_scan: bool = Field(True, description="Enable security scanning")
    enable_coverage_analysis: bool = Field(True, description="Enable coverage analysis")
    enable_quality_assessment: bool = Field(True, description="Enable quality assessment")

    @validator('project_path')
    def validate_project_path(cls, v):
        """Validate that project path exists and is a directory."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return str(path.absolute())


class SecurityScanRequest(BaseModel):
    """Request model for security scanning."""
    file_path: str = Field(..., description="Path to file to scan")
    rules: Optional[List[str]] = Field(None, description="Specific security rules to apply")
    strict_mode: bool = Field(False, description="Use strict security rules")
    include_low_severity: bool = Field(True, description="Include low severity issues")

    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        return str(path.absolute())


# Response Models
class SecurityIssueResponse(BaseModel):
    """Response model for security issues."""
    rule_id: str
    rule_name: str
    severity: SecuritySeverity
    category: str
    description: str
    file_path: str
    line_number: Optional[int]
    column_number: Optional[int]
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str]
    owasp_category: Optional[str]
    confidence: float
    false_positive: bool = False
    suppressed: bool = False


class TestCaseResponse(BaseModel):
    """Response model for generated test cases."""
    test_name: str
    test_type: str  # unit, integration, edge_case, error_path, benchmark
    function_name: str
    test_content: str
    assertions_count: int
    complexity_score: float
    execution_time_ms: Optional[int]
    passed: Optional[bool]


class AnalysisResponse(BaseModel):
    """Response model for file analysis."""
    session_id: str
    file_path: str
    language: str
    status: ProcessingStatus
    processing_time_ms: int

    # Generated artifacts
    tests_generated: Optional[str] = Field(None, description="Path to generated test file")
    test_cases: List[TestCaseResponse] = Field(default_factory=list)

    # Analysis results
    coverage_percentage: Optional[float]
    quality_score: Optional[float]
    security_issues: List[SecurityIssueResponse] = Field(default_factory=list)

    # Metadata
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ProjectMetricsResponse(BaseModel):
    """Response model for project metrics."""
    session_id: str
    project_path: str
    total_files: int
    analyzed_files: int
    generated_tests: int
    total_test_cases: int
    average_coverage: float
    average_quality_score: float

    # Security metrics
    security_issues_total: int
    security_issues_critical: int
    security_issues_high: int
    security_issues_medium: int
    security_issues_low: int

    # Performance metrics
    processing_time_total_ms: int
    processing_time_average_ms: float

    # Code metrics
    lines_of_code: int
    lines_of_tests: int
    test_to_code_ratio: float
    languages_used: List[str]
    frameworks_detected: List[str]

    calculated_at: datetime


class SessionResponse(BaseModel):
    """Response model for processing sessions."""
    session_id: str
    project_path: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: ProcessingStatus
    total_files: int
    processed_files: int
    failed_files: int
    configuration: Dict[str, Any]


class ProjectAnalysisResponse(BaseModel):
    """Response model for project analysis."""
    session: SessionResponse
    results: List[AnalysisResponse]
    metrics: ProjectMetricsResponse
    recommendations: List[str] = Field(default_factory=list)


class SecurityScanResponse(BaseModel):
    """Response model for security scanning."""
    file_path: str
    scan_time_ms: int
    total_issues: int
    issues: List[SecurityIssueResponse]
    summary: Dict[SecuritySeverity, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = "healthy"
    version: str
    timestamp: datetime
    database: Dict[str, Any] = Field(default_factory=dict)
    dependencies: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: bool = True
    message: str
    error_type: Optional[str]
    details: Optional[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str]


# Pagination models
class PaginationParams(BaseModel):
    """Parameters for paginated requests."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", pattern="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

    @validator('pages', always=True)
    def calculate_pages(cls, v, values):
        """Calculate total pages based on total items and page size."""
        total = values.get('total', 0)
        size = values.get('size', 1)
        return (total + size - 1) // size if total > 0 else 0
