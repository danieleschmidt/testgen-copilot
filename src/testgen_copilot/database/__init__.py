"""Database layer for TestGen Copilot persistence."""

from .connection import DatabaseConnection, get_database
from .migrations import MigrationManager, run_migrations
from .models import AnalysisResult, ProcessingSession, ProjectMetrics, SecurityIssue, TestCase
from .repositories import (
    AnalysisRepository,
    MetricsRepository,
    SecurityRepository,
    SessionRepository,
    TestCaseRepository,
)

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
