"""API route definitions for TestGen Copilot."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse

from .models import (
    AnalysisRequest, AnalysisResponse, ProjectAnalysisRequest, ProjectAnalysisResponse,
    SecurityScanRequest, SecurityScanResponse, HealthResponse, SessionResponse,
    ProjectMetricsResponse, PaginationParams, PaginatedResponse, ErrorResponse
)
from ..core import TestGenOrchestrator, ProcessingStatus, AnalysisPhase
from ..generator import GenerationConfig
from ..security import SecurityScanner
from ..database import (
    get_database, SessionRepository, AnalysisRepository, SecurityRepository,
    MetricsRepository, run_migrations
)
from ..logging_config import get_logger
from ..version import __version__, get_version_info


# Create routers
health_bp = APIRouter()
analysis_bp = APIRouter()
sessions_bp = APIRouter()
security_bp = APIRouter()
metrics_bp = APIRouter()

logger = get_logger("testgen_copilot.api.routes")


# Dependency functions
def get_session_repo() -> SessionRepository:
    """Get session repository dependency."""
    return SessionRepository(get_database())


def get_analysis_repo() -> AnalysisRepository:
    """Get analysis repository dependency."""
    return AnalysisRepository(get_database())


def get_security_repo() -> SecurityRepository:
    """Get security repository dependency."""
    return SecurityRepository(get_database())


def get_metrics_repo() -> MetricsRepository:
    """Get metrics repository dependency."""
    return MetricsRepository(get_database())


# Health endpoints
@health_bp.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db = get_database()
    db_info = db.get_database_info()
    
    # Check database connectivity
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        version=__version__,
        timestamp=datetime.now(timezone.utc),
        database={
            "status": db_status,
            "path": db_info.get("db_path"),
            "size_bytes": db_info.get("size_bytes", 0),
            "table_count": db_info.get("table_count", 0)
        },
        dependencies={
            "python_version": get_version_info().get("python_version"),
            "platform": get_version_info().get("platform")
        }
    )


@health_bp.get("/version", response_model=Dict[str, Any])
async def version_info():
    """Get detailed version information."""
    return get_version_info()


# Analysis endpoints
@analysis_bp.post("/file", response_model=AnalysisResponse)
async def analyze_file(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    session_repo: SessionRepository = Depends(get_session_repo),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repo)
):
    """Analyze a single source file."""
    session_id = str(uuid.uuid4())
    
    logger.info("Starting file analysis", {
        "session_id": session_id,
        "file_path": request.file_path,
        "language": request.language
    })
    
    try:
        # Create processing session
        from ..database.models import ProcessingSession
        session = ProcessingSession(
            session_id=session_id,
            project_path=str(Path(request.file_path).parent),
            status=ProcessingStatus.IN_PROGRESS,
            total_files=1,
            configuration=request.dict()
        )
        session_repo.create_session(session)
        
        # Create generation config
        config = GenerationConfig(
            language=request.language or "python",
            include_edge_cases=request.include_edge_cases,
            include_error_paths=request.include_error_paths,
            include_benchmarks=request.include_benchmarks,
            include_integration_tests=request.include_integration_tests,
            use_mocking=request.use_mocking
        )
        
        # Create orchestrator
        orchestrator = TestGenOrchestrator(
            config=config,
            enable_security=request.enable_security_scan,
            enable_coverage=request.enable_coverage_analysis,
            enable_quality=request.enable_quality_assessment
        )
        
        # Process file
        phases = request.phases or list(AnalysisPhase)
        result = await orchestrator.process_file(
            request.file_path,
            request.output_dir,
            phases=phases
        )
        
        # Update session
        session.status = ProcessingStatus.COMPLETED if result.status == ProcessingStatus.COMPLETED else ProcessingStatus.FAILED
        session.processed_files = 1 if result.status == ProcessingStatus.COMPLETED else 0
        session.failed_files = 1 if result.status == ProcessingStatus.FAILED else 0
        session.completed_at = datetime.now(timezone.utc)
        session_repo.update_session(session)
        
        # Convert to response model
        return AnalysisResponse(
            session_id=session_id,
            file_path=result.file_path,
            language=config.language,
            status=result.status,
            processing_time_ms=int(result.processing_time * 1000),
            tests_generated=str(result.tests_generated) if result.tests_generated else None,
            coverage_percentage=result.coverage_result.percentage if result.coverage_result else None,
            quality_score=result.quality_score,
            security_issues=[],  # TODO: Convert security issues
            errors=result.errors,
            warnings=result.warnings,
            metadata={"phases_completed": [p.value for p in phases]},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error("File analysis failed", {
            "session_id": session_id,
            "file_path": request.file_path,
            "error": str(e)
        })
        
        # Update session status
        try:
            session.status = ProcessingStatus.FAILED
            session.failed_files = 1
            session.completed_at = datetime.now(timezone.utc)
            session_repo.update_session(session)
        except:
            pass  # Don't fail the error response if session update fails
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@analysis_bp.post("/project", response_model=ProjectAnalysisResponse)
async def analyze_project(
    request: ProjectAnalysisRequest,
    background_tasks: BackgroundTasks,
    session_repo: SessionRepository = Depends(get_session_repo),
    analysis_repo: AnalysisRepository = Depends(get_analysis_repo),
    metrics_repo: MetricsRepository = Depends(get_metrics_repo)
):
    """Analyze an entire project."""
    session_id = str(uuid.uuid4())
    
    logger.info("Starting project analysis", {
        "session_id": session_id,
        "project_path": request.project_path,
        "concurrent_limit": request.concurrent_limit
    })
    
    try:
        # Create processing session
        from ..database.models import ProcessingSession
        session = ProcessingSession(
            session_id=session_id,
            project_path=request.project_path,
            status=ProcessingStatus.IN_PROGRESS,
            configuration=request.dict()
        )
        session_repo.create_session(session)
        
        # Create generation config
        config = GenerationConfig(
            include_edge_cases=request.include_edge_cases,
            include_error_paths=request.include_error_paths,
            include_benchmarks=request.include_benchmarks,
            include_integration_tests=request.include_integration_tests,
            use_mocking=request.use_mocking
        )
        
        # Create orchestrator
        orchestrator = TestGenOrchestrator(
            config=config,
            enable_security=request.enable_security_scan,
            enable_coverage=request.enable_coverage_analysis,
            enable_quality=request.enable_quality_assessment,
            concurrent_limit=request.concurrent_limit
        )
        
        # Process project
        results = await orchestrator.process_project(
            request.project_path,
            request.output_dir,
            file_patterns=request.file_patterns,
            exclude_patterns=request.exclude_patterns
        )
        
        # Generate comprehensive report
        report = orchestrator.generate_comprehensive_report(results)
        
        # Update session
        successful_count = sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED)
        failed_count = len(results) - successful_count
        
        session.status = ProcessingStatus.COMPLETED
        session.total_files = len(results)
        session.processed_files = successful_count
        session.failed_files = failed_count
        session.completed_at = datetime.now(timezone.utc)
        session_repo.update_session(session)
        
        # Convert results to response models
        analysis_responses = []
        for file_path, result in results.items():
            analysis_responses.append(AnalysisResponse(
                session_id=session_id,
                file_path=file_path,
                language="python",  # TODO: Detect language
                status=result.status,
                processing_time_ms=int(result.processing_time * 1000),
                tests_generated=str(result.tests_generated) if result.tests_generated else None,
                coverage_percentage=result.coverage_result.percentage if result.coverage_result else None,
                quality_score=result.quality_score,
                security_issues=[],  # TODO: Convert security issues
                errors=result.errors,
                warnings=result.warnings,
                metadata={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ))
        
        # Create metrics response
        metrics_response = ProjectMetricsResponse(
            session_id=session_id,
            project_path=request.project_path,
            total_files=report["metrics"]["files_analyzed"],
            analyzed_files=successful_count,
            generated_tests=report["metrics"]["tests_generated"],
            total_test_cases=0,  # TODO: Count test cases
            average_coverage=report["metrics"]["average_coverage_percentage"],
            average_quality_score=report["metrics"]["average_quality_score"],
            security_issues_total=report["metrics"]["security_issues_found"],
            security_issues_critical=report["security_summary"]["critical_issues"],
            security_issues_high=report["security_summary"]["high_issues"],
            security_issues_medium=report["security_summary"]["medium_issues"],
            security_issues_low=report["security_summary"]["low_issues"],
            processing_time_total_ms=int(report["metrics"]["processing_time_seconds"] * 1000),
            processing_time_average_ms=int(report["metrics"]["processing_time_seconds"] * 1000 / max(len(results), 1)),
            lines_of_code=0,  # TODO: Calculate
            lines_of_tests=0,  # TODO: Calculate
            test_to_code_ratio=0.0,  # TODO: Calculate
            languages_used=["python"],  # TODO: Detect languages
            frameworks_detected=[],  # TODO: Detect frameworks
            calculated_at=datetime.now(timezone.utc)
        )
        
        return ProjectAnalysisResponse(
            session=SessionResponse(
                session_id=session_id,
                project_path=request.project_path,
                started_at=session.started_at,
                completed_at=session.completed_at,
                status=session.status,
                total_files=session.total_files,
                processed_files=session.processed_files,
                failed_files=session.failed_files,
                configuration=session.configuration
            ),
            results=analysis_responses,
            metrics=metrics_response,
            recommendations=report["recommendations"]
        )
        
    except Exception as e:
        logger.error("Project analysis failed", {
            "session_id": session_id,
            "project_path": request.project_path,
            "error": str(e)
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Project analysis failed: {str(e)}"
        )


@analysis_bp.get("/download/{session_id}")
async def download_tests(session_id: str):
    """Download generated tests for a session."""
    # TODO: Implement test file download
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Test download not yet implemented"
    )


# Session management endpoints
@sessions_bp.get("/", response_model=PaginatedResponse)
async def list_sessions(
    pagination: PaginationParams = Depends(),
    project_path: Optional[str] = Query(None, description="Filter by project path"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    session_repo: SessionRepository = Depends(get_session_repo)
):
    """List processing sessions with pagination."""
    try:
        status_enum = ProcessingStatus(status_filter) if status_filter else None
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {status_filter}"
        )
    
    sessions = session_repo.list_sessions(
        project_path=project_path,
        status=status_enum,
        limit=pagination.size * pagination.page  # Simple limit for now
    )
    
    # Convert to response models
    session_responses = []
    for session in sessions:
        session_responses.append(SessionResponse(
            session_id=session.session_id,
            project_path=session.project_path,
            started_at=session.started_at,
            completed_at=session.completed_at,
            status=session.status,
            total_files=session.total_files,
            processed_files=session.processed_files,
            failed_files=session.failed_files,
            configuration=session.configuration
        ))
    
    return PaginatedResponse(
        items=session_responses,
        total=len(session_responses),
        page=pagination.page,
        size=pagination.size,
        pages=(len(session_responses) + pagination.size - 1) // pagination.size
    )


@sessions_bp.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_repo: SessionRepository = Depends(get_session_repo)
):
    """Get session details by ID."""
    session = session_repo.get_session_by_id(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    return SessionResponse(
        session_id=session.session_id,
        project_path=session.project_path,
        started_at=session.started_at,
        completed_at=session.completed_at,
        status=session.status,
        total_files=session.total_files,
        processed_files=session.processed_files,
        failed_files=session.failed_files,
        configuration=session.configuration
    )


@sessions_bp.get("/{session_id}/results", response_model=List[AnalysisResponse])
async def get_session_results(
    session_id: str,
    analysis_repo: AnalysisRepository = Depends(get_analysis_repo)
):
    """Get analysis results for a session."""
    results = analysis_repo.get_results_by_session(session_id)
    
    # Convert to response models
    response_list = []
    for result in results:
        response_list.append(AnalysisResponse(
            session_id=result.session_id,
            file_path=result.file_path,
            language=result.language,
            status=result.status,
            processing_time_ms=result.processing_time_ms,
            tests_generated=result.tests_generated,
            coverage_percentage=result.coverage_percentage,
            quality_score=result.quality_score,
            security_issues=[],  # TODO: Load security issues
            errors=result.errors,
            warnings=result.warnings,
            metadata=result.metadata,
            created_at=result.created_at,
            updated_at=result.updated_at
        ))
    
    return response_list


# Security endpoints
@security_bp.post("/scan", response_model=SecurityScanResponse)
async def scan_file(request: SecurityScanRequest):
    """Perform security scan on a file."""
    logger.info("Starting security scan", {
        "file_path": request.file_path,
        "strict_mode": request.strict_mode
    })
    
    try:
        scanner = SecurityScanner()
        report = scanner.scan_file(request.file_path)
        
        # Convert to response model
        from .models import SecurityIssueResponse
        issues = []
        for issue in report.issues:
            issues.append(SecurityIssueResponse(
                rule_id=issue.rule_id,
                rule_name=issue.rule_name,
                severity=issue.severity,
                category=issue.category,
                description=issue.description,
                file_path=issue.file_path,
                line_number=issue.line_number,
                column_number=issue.column_number,
                code_snippet=issue.code_snippet,
                recommendation=issue.recommendation,
                cwe_id=issue.cwe_id,
                owasp_category=issue.owasp_category,
                confidence=issue.confidence
            ))
        
        # Calculate summary
        summary = {}
        for issue in issues:
            severity = issue.severity
            summary[severity] = summary.get(severity, 0) + 1
        
        return SecurityScanResponse(
            file_path=request.file_path,
            scan_time_ms=int(report.scan_time * 1000),
            total_issues=len(issues),
            issues=issues,
            summary=summary
        )
        
    except Exception as e:
        logger.error("Security scan failed", {
            "file_path": request.file_path,
            "error": str(e)
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security scan failed: {str(e)}"
        )


# Metrics endpoints
@metrics_bp.get("/{session_id}", response_model=ProjectMetricsResponse)
async def get_session_metrics(
    session_id: str,
    metrics_repo: MetricsRepository = Depends(get_metrics_repo)
):
    """Get metrics for a processing session."""
    metrics = metrics_repo.get_metrics_by_session(session_id)
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics not found for session: {session_id}"
        )
    
    return ProjectMetricsResponse(
        session_id=metrics.session_id,
        project_path=metrics.project_path,
        total_files=metrics.total_files,
        analyzed_files=metrics.analyzed_files,
        generated_tests=metrics.generated_tests,
        total_test_cases=metrics.total_test_cases,
        average_coverage=metrics.average_coverage,
        average_quality_score=metrics.average_quality_score,
        security_issues_total=metrics.security_issues_total,
        security_issues_critical=metrics.security_issues_critical,
        security_issues_high=metrics.security_issues_high,
        security_issues_medium=metrics.security_issues_medium,
        security_issues_low=metrics.security_issues_low,
        processing_time_total_ms=metrics.processing_time_total_ms,
        processing_time_average_ms=metrics.processing_time_average_ms,
        lines_of_code=metrics.lines_of_code,
        lines_of_tests=metrics.lines_of_tests,
        test_to_code_ratio=metrics.test_to_code_ratio,
        languages_used=metrics.languages_used,
        frameworks_detected=metrics.frameworks_detected,
        calculated_at=metrics.calculated_at
    )