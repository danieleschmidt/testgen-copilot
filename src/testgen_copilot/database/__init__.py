"""Database layer for TestGen Copilot persistence."""

from .connection import DatabaseConnection, get_database
from .models import (
    AnalysisResult,
    TestCase,
    SecurityIssue,
    ProjectMetrics,
    ProcessingSession
)
from .repositories import (
    AnalysisRepository,
    TestCaseRepository,
    SecurityRepository,
    MetricsRepository,
    SessionRepository
)
from .migrations import MigrationManager, run_migrations

__all__ = [
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
    "MigrationManager",
    "run_migrations"
]