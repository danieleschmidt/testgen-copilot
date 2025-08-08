"""Database migration management for TestGen Copilot."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..logging_config import get_database_logger
from .connection import DatabaseConnection, get_database


class Migration:
    """Represents a single database migration."""

    def __init__(self, version: int, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql

    def __str__(self) -> str:
        return f"Migration {self.version:03d}: {self.name}"


class MigrationManager:
    """Manages database schema migrations."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = get_database_logger()
        self.migrations = self._get_migrations()

    def _get_migrations(self) -> List[Migration]:
        """Define all database migrations."""
        return [
            Migration(
                version=1,
                name="create_sessions_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS processing_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    project_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    configuration TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON processing_sessions(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_project_path ON processing_sessions(project_path);
                CREATE INDEX IF NOT EXISTS idx_sessions_status ON processing_sessions(status);
                """
            ),

            Migration(
                version=2,
                name="create_analysis_results_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    processing_time_ms INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    tests_generated TEXT,
                    coverage_percentage REAL,
                    quality_score REAL,
                    security_issues_count INTEGER DEFAULT 0,
                    errors TEXT DEFAULT '[]',
                    warnings TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES processing_sessions(session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_analysis_session_id ON analysis_results(session_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_file_path ON analysis_results(file_path);
                CREATE INDEX IF NOT EXISTS idx_analysis_language ON analysis_results(language);
                CREATE INDEX IF NOT EXISTS idx_analysis_status ON analysis_results(status);
                """
            ),

            Migration(
                version=3,
                name="create_test_cases_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_result_id INTEGER NOT NULL,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL DEFAULT 'unit',
                    function_name TEXT NOT NULL,
                    test_content TEXT NOT NULL,
                    assertions_count INTEGER DEFAULT 0,
                    complexity_score REAL DEFAULT 0.0,
                    execution_time_ms INTEGER,
                    passed BOOLEAN,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_result_id) REFERENCES analysis_results(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_tests_analysis_id ON test_cases(analysis_result_id);
                CREATE INDEX IF NOT EXISTS idx_tests_type ON test_cases(test_type);
                CREATE INDEX IF NOT EXISTS idx_tests_function ON test_cases(function_name);
                CREATE INDEX IF NOT EXISTS idx_tests_passed ON test_cases(passed);
                """
            ),

            Migration(
                version=4,
                name="create_security_issues_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS security_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_result_id INTEGER NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'low',
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    column_number INTEGER,
                    code_snippet TEXT DEFAULT '',
                    recommendation TEXT DEFAULT '',
                    cwe_id TEXT,
                    owasp_category TEXT,
                    confidence REAL DEFAULT 0.0,
                    false_positive BOOLEAN DEFAULT FALSE,
                    suppressed BOOLEAN DEFAULT FALSE,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_result_id) REFERENCES analysis_results(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_security_analysis_id ON security_issues(analysis_result_id);
                CREATE INDEX IF NOT EXISTS idx_security_rule_id ON security_issues(rule_id);
                CREATE INDEX IF NOT EXISTS idx_security_severity ON security_issues(severity);
                CREATE INDEX IF NOT EXISTS idx_security_category ON security_issues(category);
                CREATE INDEX IF NOT EXISTS idx_security_file_path ON security_issues(file_path);
                CREATE INDEX IF NOT EXISTS idx_security_false_positive ON security_issues(false_positive);
                """
            ),

            Migration(
                version=5,
                name="create_project_metrics_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS project_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    project_path TEXT NOT NULL,
                    total_files INTEGER DEFAULT 0,
                    analyzed_files INTEGER DEFAULT 0,
                    generated_tests INTEGER DEFAULT 0,
                    total_test_cases INTEGER DEFAULT 0,
                    average_coverage REAL DEFAULT 0.0,
                    average_quality_score REAL DEFAULT 0.0,
                    security_issues_total INTEGER DEFAULT 0,
                    security_issues_critical INTEGER DEFAULT 0,
                    security_issues_high INTEGER DEFAULT 0,
                    security_issues_medium INTEGER DEFAULT 0,
                    security_issues_low INTEGER DEFAULT 0,
                    processing_time_total_ms INTEGER DEFAULT 0,
                    processing_time_average_ms REAL DEFAULT 0.0,
                    lines_of_code INTEGER DEFAULT 0,
                    lines_of_tests INTEGER DEFAULT 0,
                    test_to_code_ratio REAL DEFAULT 0.0,
                    languages_used TEXT DEFAULT '[]',
                    frameworks_detected TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES processing_sessions(session_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON project_metrics(session_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_project_path ON project_metrics(project_path);
                CREATE INDEX IF NOT EXISTS idx_metrics_calculated_at ON project_metrics(calculated_at);
                """
            ),

            Migration(
                version=6,
                name="create_migration_history_table",
                up_sql="""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER DEFAULT 0
                );
                
                CREATE INDEX IF NOT EXISTS idx_migration_version ON migration_history(version);
                """
            )
        ]

    def get_current_version(self) -> int:
        """Get the current database schema version."""
        try:
            row = self.db.fetch_one("PRAGMA user_version")
            return row[0] if row else 0
        except Exception:
            return 0

    def set_version(self, version: int) -> None:
        """Set the database schema version."""
        self.db.execute_query(f"PRAGMA user_version = {version}")

    def get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        try:
            rows = self.db.fetch_all(
                "SELECT version FROM migration_history ORDER BY version"
            )
            return [row["version"] for row in rows]
        except Exception:
            # Migration history table doesn't exist yet
            return []

    def record_migration(self, migration: Migration, execution_time_ms: int) -> None:
        """Record a migration in the history table."""
        try:
            self.db.execute_query(
                """
                INSERT OR REPLACE INTO migration_history 
                (version, name, applied_at, execution_time_ms)
                VALUES (?, ?, ?, ?)
                """,
                (
                    migration.version,
                    migration.name,
                    datetime.now(timezone.utc).isoformat(),
                    execution_time_ms
                )
            )
        except Exception as e:
            self.logger.warning("Could not record migration history", {
                "migration": str(migration),
                "error": str(e)
            })

    def migrate_up(self, target_version: Optional[int] = None) -> int:
        """Run pending migrations up to target version."""
        current_version = self.get_current_version()
        applied_migrations = set(self.get_applied_migrations())

        if target_version is None:
            target_version = max(m.version for m in self.migrations)

        migrations_to_run = [
            m for m in self.migrations
            if m.version <= target_version and m.version not in applied_migrations
        ]

        migrations_to_run.sort(key=lambda m: m.version)

        if not migrations_to_run:
            self.logger.info("No pending migrations to run", {
                "current_version": current_version,
                "target_version": target_version
            })
            return current_version

        self.logger.info("Starting database migrations", {
            "current_version": current_version,
            "target_version": target_version,
            "migrations_to_run": len(migrations_to_run)
        })

        for migration in migrations_to_run:
            start_time = datetime.now()

            try:
                self.logger.info("Applying migration", {
                    "migration": str(migration)
                })

                # Execute migration SQL
                for statement in migration.up_sql.strip().split(';'):
                    statement = statement.strip()
                    if statement:
                        self.db.execute_query(statement)

                # Record execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_migration(migration, int(execution_time))

                # Update schema version
                self.set_version(migration.version)

                self.logger.info("Migration applied successfully", {
                    "migration": str(migration),
                    "execution_time_ms": int(execution_time)
                })

            except Exception as e:
                self.logger.error("Migration failed", {
                    "migration": str(migration),
                    "error": str(e)
                })
                raise RuntimeError(f"Migration {migration.version} failed: {e}")

        final_version = self.get_current_version()
        self.logger.info("Database migrations completed", {
            "final_version": final_version,
            "migrations_applied": len(migrations_to_run)
        })

        return final_version

    def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status information."""
        current_version = self.get_current_version()
        applied_migrations = set(self.get_applied_migrations())

        latest_version = max(m.version for m in self.migrations) if self.migrations else 0

        pending_migrations = [
            m for m in self.migrations
            if m.version not in applied_migrations
        ]

        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "is_up_to_date": len(pending_migrations) == 0,
            "pending_migrations": len(pending_migrations),
            "applied_migrations": len(applied_migrations),
            "total_migrations": len(self.migrations),
            "pending_migration_details": [
                {"version": m.version, "name": m.name}
                for m in sorted(pending_migrations, key=lambda x: x.version)
            ]
        }


def run_migrations(db_path: Optional[Path] = None) -> int:
    """Run all pending database migrations."""
    db = get_database(db_path)
    manager = MigrationManager(db)
    return manager.migrate_up()
