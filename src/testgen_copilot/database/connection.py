"""Database connection management for TestGen Copilot."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, Optional

from ..logging_config import get_database_logger


class DatabaseConnection:
    """Thread-safe SQLite database connection manager."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self._get_default_db_path()
        self.logger = get_database_logger()
        self._connection_cache: Dict[int, sqlite3.Connection] = {}
        self._lock = Lock()

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database on first connection
        self._initialize_database()

    def _get_default_db_path(self) -> Path:
        """Get default database path from environment or user home."""
        data_dir = os.getenv('TESTGEN_DATA_DIR', '~/.testgen')
        data_path = Path(data_dir).expanduser()
        return data_path / 'testgen.db'

    def _initialize_database(self) -> None:
        """Initialize database with basic configuration."""
        with self.get_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Set journal mode for better concurrent access
            conn.execute("PRAGMA journal_mode = WAL")

            # Set synchronous mode for performance
            conn.execute("PRAGMA synchronous = NORMAL")

            self.logger.info("Database initialized", {
                "db_path": str(self.db_path),
                "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
            })

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with automatic transaction management."""
        thread_id = os.getpid()  # Use process ID for connection caching

        with self._lock:
            if thread_id not in self._connection_cache:
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False
                )

                # Configure connection
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                conn.execute("PRAGMA foreign_keys = ON")

                self._connection_cache[thread_id] = conn

                self.logger.debug("Created new database connection", {
                    "thread_id": thread_id,
                    "db_path": str(self.db_path)
                })

            conn = self._connection_cache[thread_id]

        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error("Database transaction failed", {
                "error": str(e),
                "thread_id": thread_id
            })
            raise

    def execute_query(self, query: str, params: Optional[tuple] = None) -> sqlite3.Cursor:
        """Execute a single query and return cursor."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            return cursor

    def execute_many(self, query: str, params_list: list) -> None:
        """Execute a query multiple times with different parameters."""
        with self.get_connection() as conn:
            conn.executemany(query, params_list)

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[sqlite3.Row]:
        """Execute query and fetch one result."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            return cursor.fetchone()

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list[sqlite3.Row]:
        """Execute query and fetch all results."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            return cursor.fetchall()

    def close_all_connections(self) -> None:
        """Close all cached connections."""
        with self._lock:
            for thread_id, conn in self._connection_cache.items():
                try:
                    conn.close()
                    self.logger.debug("Closed database connection", {
                        "thread_id": thread_id
                    })
                except Exception as e:
                    self.logger.warning("Error closing database connection", {
                        "thread_id": thread_id,
                        "error": str(e)
                    })

            self._connection_cache.clear()

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        info = {
            "db_path": str(self.db_path),
            "exists": self.db_path.exists(),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "active_connections": len(self._connection_cache)
        }

        if self.db_path.exists():
            try:
                with self.get_connection() as conn:
                    # Get table count
                    cursor = conn.execute(
                        "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'"
                    )
                    info["table_count"] = cursor.fetchone()["count"]

                    # Get database version
                    cursor = conn.execute("PRAGMA user_version")
                    info["schema_version"] = cursor.fetchone()[0]

                    # Get page size and page count
                    cursor = conn.execute("PRAGMA page_size")
                    info["page_size"] = cursor.fetchone()[0]

                    cursor = conn.execute("PRAGMA page_count")
                    info["page_count"] = cursor.fetchone()[0]

            except Exception as e:
                self.logger.warning("Could not retrieve database info", {
                    "error": str(e)
                })
                info["error"] = str(e)

        return info


# Global database instance
_database_instance: Optional[DatabaseConnection] = None
_database_lock = Lock()


def get_database(db_path: Optional[Path] = None) -> DatabaseConnection:
    """Get the global database instance (singleton pattern)."""
    global _database_instance

    with _database_lock:
        if _database_instance is None:
            _database_instance = DatabaseConnection(db_path)
        return _database_instance


def close_database() -> None:
    """Close the global database instance."""
    global _database_instance

    with _database_lock:
        if _database_instance:
            _database_instance.close_all_connections()
            _database_instance = None
