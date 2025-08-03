"""Repository layer for TestGen Copilot data access."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from .connection import DatabaseConnection
from .models import (
    ProcessingSession,
    AnalysisResult,
    TestCase,
    SecurityIssue,
    ProjectMetrics,
    ProcessingStatus,
    SecuritySeverity
)
from ..logging_config import get_database_logger


class BaseRepository(ABC):
    """Base repository with common functionality."""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = get_database_logger()
    
    @staticmethod
    def _serialize_json(data: Any) -> str:
        """Serialize data to JSON string."""
        if isinstance(data, (list, dict)):
            return json.dumps(data)
        return str(data)
    
    @staticmethod
    def _deserialize_json(data: str, default: Any = None) -> Any:
        """Deserialize JSON string to data."""
        if not data:
            return default or []
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return default or []


class SessionRepository(BaseRepository):
    """Repository for managing processing sessions."""
    
    def create_session(self, session: ProcessingSession) -> ProcessingSession:
        """Create a new processing session."""
        cursor = self.db.execute_query(
            """
            INSERT INTO processing_sessions 
            (session_id, project_path, started_at, status, total_files, 
             processed_files, failed_files, configuration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.project_path,
                session.started_at.isoformat(),
                session.status.value,
                session.total_files,
                session.processed_files,
                session.failed_files,
                self._serialize_json(session.configuration)
            )
        )
        
        session.id = cursor.lastrowid
        
        self.logger.info("Created processing session", {
            "session_id": session.session_id,
            "project_path": session.project_path,
            "id": session.id
        })
        
        return session
    
    def get_session_by_id(self, session_id: str) -> Optional[ProcessingSession]:
        """Get session by session ID."""
        row = self.db.fetch_one(
            "SELECT * FROM processing_sessions WHERE session_id = ?",
            (session_id,)
        )
        
        if not row:
            return None
        
        return ProcessingSession(
            id=row["id"],
            session_id=row["session_id"],
            project_path=row["project_path"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            status=ProcessingStatus(row["status"]),
            total_files=row["total_files"],
            processed_files=row["processed_files"],
            failed_files=row["failed_files"],
            configuration=self._deserialize_json(row["configuration"], {})
        )
    
    def update_session(self, session: ProcessingSession) -> ProcessingSession:
        """Update an existing session."""
        self.db.execute_query(
            """
            UPDATE processing_sessions 
            SET completed_at = ?, status = ?, total_files = ?, 
                processed_files = ?, failed_files = ?, configuration = ?
            WHERE session_id = ?
            """,
            (
                session.completed_at.isoformat() if session.completed_at else None,
                session.status.value,
                session.total_files,
                session.processed_files,
                session.failed_files,
                self._serialize_json(session.configuration),
                session.session_id
            )
        )
        
        self.logger.debug("Updated processing session", {
            "session_id": session.session_id,
            "status": session.status.value
        })
        
        return session
    
    def list_sessions(
        self,
        project_path: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100
    ) -> List[ProcessingSession]:
        """List processing sessions with optional filters."""
        query = "SELECT * FROM processing_sessions WHERE 1=1"
        params = []
        
        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self.db.fetch_all(query, tuple(params))
        
        sessions = []
        for row in rows:
            sessions.append(ProcessingSession(
                id=row["id"],
                session_id=row["session_id"],
                project_path=row["project_path"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                status=ProcessingStatus(row["status"]),
                total_files=row["total_files"],
                processed_files=row["processed_files"],
                failed_files=row["failed_files"],
                configuration=self._deserialize_json(row["configuration"], {})
            ))
        
        return sessions


class AnalysisRepository(BaseRepository):
    """Repository for managing analysis results."""
    
    def create_result(self, result: AnalysisResult) -> AnalysisResult:
        """Create a new analysis result."""
        cursor = self.db.execute_query(
            """
            INSERT INTO analysis_results 
            (session_id, file_path, language, processing_time_ms, status,
             tests_generated, coverage_percentage, quality_score, 
             security_issues_count, errors, warnings, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.session_id,
                result.file_path,
                result.language,
                result.processing_time_ms,
                result.status.value,
                result.tests_generated,
                result.coverage_percentage,
                result.quality_score,
                result.security_issues_count,
                self._serialize_json(result.errors),
                self._serialize_json(result.warnings),
                self._serialize_json(result.metadata)
            )
        )
        
        result.id = cursor.lastrowid
        
        self.logger.debug("Created analysis result", {
            "id": result.id,
            "session_id": result.session_id,
            "file_path": result.file_path
        })
        
        return result
    
    def update_result(self, result: AnalysisResult) -> AnalysisResult:
        """Update an existing analysis result."""
        result.updated_at = datetime.now(timezone.utc)
        
        self.db.execute_query(
            """
            UPDATE analysis_results 
            SET processing_time_ms = ?, status = ?, tests_generated = ?,
                coverage_percentage = ?, quality_score = ?, 
                security_issues_count = ?, errors = ?, warnings = ?,
                metadata = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                result.processing_time_ms,
                result.status.value,
                result.tests_generated,
                result.coverage_percentage,
                result.quality_score,
                result.security_issues_count,
                self._serialize_json(result.errors),
                self._serialize_json(result.warnings),
                self._serialize_json(result.metadata),
                result.updated_at.isoformat(),
                result.id
            )
        )
        
        return result
    
    def get_results_by_session(self, session_id: str) -> List[AnalysisResult]:
        """Get all analysis results for a session."""
        rows = self.db.fetch_all(
            "SELECT * FROM analysis_results WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        
        results = []
        for row in rows:
            results.append(AnalysisResult(
                id=row["id"],
                session_id=row["session_id"],
                file_path=row["file_path"],
                language=row["language"],
                processing_time_ms=row["processing_time_ms"],
                status=ProcessingStatus(row["status"]),
                tests_generated=row["tests_generated"],
                coverage_percentage=row["coverage_percentage"],
                quality_score=row["quality_score"],
                security_issues_count=row["security_issues_count"],
                errors=self._deserialize_json(row["errors"], []),
                warnings=self._deserialize_json(row["warnings"], []),
                metadata=self._deserialize_json(row["metadata"], {}),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            ))
        
        return results


class TestCaseRepository(BaseRepository):
    """Repository for managing test cases."""
    
    def create_test_case(self, test_case: TestCase) -> TestCase:
        """Create a new test case."""
        cursor = self.db.execute_query(
            """
            INSERT INTO test_cases 
            (analysis_result_id, test_name, test_type, function_name,
             test_content, assertions_count, complexity_score,
             execution_time_ms, passed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test_case.analysis_result_id,
                test_case.test_name,
                test_case.test_type,
                test_case.function_name,
                test_case.test_content,
                test_case.assertions_count,
                test_case.complexity_score,
                test_case.execution_time_ms,
                test_case.passed,
                self._serialize_json(test_case.metadata)
            )
        )
        
        test_case.id = cursor.lastrowid
        return test_case
    
    def get_test_cases_by_analysis(self, analysis_result_id: int) -> List[TestCase]:
        """Get all test cases for an analysis result."""
        rows = self.db.fetch_all(
            "SELECT * FROM test_cases WHERE analysis_result_id = ? ORDER BY test_name",
            (analysis_result_id,)
        )
        
        test_cases = []
        for row in rows:
            test_cases.append(TestCase(
                id=row["id"],
                analysis_result_id=row["analysis_result_id"],
                test_name=row["test_name"],
                test_type=row["test_type"],
                function_name=row["function_name"],
                test_content=row["test_content"],
                assertions_count=row["assertions_count"],
                complexity_score=row["complexity_score"],
                execution_time_ms=row["execution_time_ms"],
                passed=row["passed"],
                metadata=self._deserialize_json(row["metadata"], {}),
                created_at=datetime.fromisoformat(row["created_at"])
            ))
        
        return test_cases


class SecurityRepository(BaseRepository):
    """Repository for managing security issues."""
    
    def create_security_issue(self, issue: SecurityIssue) -> SecurityIssue:
        """Create a new security issue."""
        cursor = self.db.execute_query(
            """
            INSERT INTO security_issues 
            (analysis_result_id, rule_id, rule_name, severity, category,
             description, file_path, line_number, column_number,
             code_snippet, recommendation, cwe_id, owasp_category,
             confidence, false_positive, suppressed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue.analysis_result_id,
                issue.rule_id,
                issue.rule_name,
                issue.severity.value,
                issue.category,
                issue.description,
                issue.file_path,
                issue.line_number,
                issue.column_number,
                issue.code_snippet,
                issue.recommendation,
                issue.cwe_id,
                issue.owasp_category,
                issue.confidence,
                issue.false_positive,
                issue.suppressed,
                self._serialize_json(issue.metadata)
            )
        )
        
        issue.id = cursor.lastrowid
        return issue
    
    def get_issues_by_analysis(self, analysis_result_id: int) -> List[SecurityIssue]:
        """Get all security issues for an analysis result."""
        rows = self.db.fetch_all(
            """
            SELECT * FROM security_issues 
            WHERE analysis_result_id = ? 
            ORDER BY severity DESC, line_number ASC
            """,
            (analysis_result_id,)
        )
        
        issues = []
        for row in rows:
            issues.append(SecurityIssue(
                id=row["id"],
                analysis_result_id=row["analysis_result_id"],
                rule_id=row["rule_id"],
                rule_name=row["rule_name"],
                severity=SecuritySeverity(row["severity"]),
                category=row["category"],
                description=row["description"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                column_number=row["column_number"],
                code_snippet=row["code_snippet"],
                recommendation=row["recommendation"],
                cwe_id=row["cwe_id"],
                owasp_category=row["owasp_category"],
                confidence=row["confidence"],
                false_positive=bool(row["false_positive"]),
                suppressed=bool(row["suppressed"]),
                metadata=self._deserialize_json(row["metadata"], {}),
                created_at=datetime.fromisoformat(row["created_at"])
            ))
        
        return issues
    
    def mark_false_positive(self, issue_id: int, false_positive: bool = True) -> None:
        """Mark a security issue as false positive."""
        self.db.execute_query(
            "UPDATE security_issues SET false_positive = ? WHERE id = ?",
            (false_positive, issue_id)
        )
    
    def suppress_issue(self, issue_id: int, suppressed: bool = True) -> None:
        """Suppress a security issue."""
        self.db.execute_query(
            "UPDATE security_issues SET suppressed = ? WHERE id = ?",
            (suppressed, issue_id)
        )


class MetricsRepository(BaseRepository):
    """Repository for managing project metrics."""
    
    def create_metrics(self, metrics: ProjectMetrics) -> ProjectMetrics:
        """Create new project metrics."""
        cursor = self.db.execute_query(
            """
            INSERT INTO project_metrics 
            (session_id, project_path, total_files, analyzed_files,
             generated_tests, total_test_cases, average_coverage,
             average_quality_score, security_issues_total,
             security_issues_critical, security_issues_high,
             security_issues_medium, security_issues_low,
             processing_time_total_ms, processing_time_average_ms,
             lines_of_code, lines_of_tests, test_to_code_ratio,
             languages_used, frameworks_detected, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.session_id,
                metrics.project_path,
                metrics.total_files,
                metrics.analyzed_files,
                metrics.generated_tests,
                metrics.total_test_cases,
                metrics.average_coverage,
                metrics.average_quality_score,
                metrics.security_issues_total,
                metrics.security_issues_critical,
                metrics.security_issues_high,
                metrics.security_issues_medium,
                metrics.security_issues_low,
                metrics.processing_time_total_ms,
                metrics.processing_time_average_ms,
                metrics.lines_of_code,
                metrics.lines_of_tests,
                metrics.test_to_code_ratio,
                self._serialize_json(metrics.languages_used),
                self._serialize_json(metrics.frameworks_detected),
                self._serialize_json(metrics.metadata)
            )
        )
        
        metrics.id = cursor.lastrowid
        return metrics
    
    def get_metrics_by_session(self, session_id: str) -> Optional[ProjectMetrics]:
        """Get project metrics for a session."""
        row = self.db.fetch_one(
            "SELECT * FROM project_metrics WHERE session_id = ?",
            (session_id,)
        )
        
        if not row:
            return None
        
        return ProjectMetrics(
            id=row["id"],
            session_id=row["session_id"],
            project_path=row["project_path"],
            total_files=row["total_files"],
            analyzed_files=row["analyzed_files"],
            generated_tests=row["generated_tests"],
            total_test_cases=row["total_test_cases"],
            average_coverage=row["average_coverage"],
            average_quality_score=row["average_quality_score"],
            security_issues_total=row["security_issues_total"],
            security_issues_critical=row["security_issues_critical"],
            security_issues_high=row["security_issues_high"],
            security_issues_medium=row["security_issues_medium"],
            security_issues_low=row["security_issues_low"],
            processing_time_total_ms=row["processing_time_total_ms"],
            processing_time_average_ms=row["processing_time_average_ms"],
            lines_of_code=row["lines_of_code"],
            lines_of_tests=row["lines_of_tests"],
            test_to_code_ratio=row["test_to_code_ratio"],
            languages_used=self._deserialize_json(row["languages_used"], []),
            frameworks_detected=self._deserialize_json(row["frameworks_detected"], []),
            metadata=self._deserialize_json(row["metadata"], {}),
            calculated_at=datetime.fromisoformat(row["calculated_at"])
        )