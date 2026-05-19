"""
app/memory/db.py

SQLite connection management for the memory system.

MemoryDB wraps a sqlite3 connection with:
  - Automatic schema creation on first open
  - WAL mode for better concurrent read performance
  - A context manager for transactional writes
  - An in-memory mode for tests (:memory:)

All other memory modules (reader, writer) accept a MemoryDB instance
so they can share one connection across the pipeline.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.memory.schema import ALL_DDL

logger = logging.getLogger(__name__)


class MemoryDB:
    """
    Manages a single SQLite connection for the memory system.

    Thread safety
    -------------
    SQLite connections are not thread-safe by default. This class uses
    check_same_thread=False + an internal lock for write operations.
    Reads are safe without the lock (WAL mode allows concurrent reads).

    Usage
    -----
    db = MemoryDB("./data/memory.db")
    db.open()

    with db.transaction() as conn:
        conn.execute("INSERT INTO ...")

    db.close()

    # Or as context manager:
    with MemoryDB(":memory:") as db:
        ...
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Parameters
        ----------
        db_path : str
            Path to the SQLite file, or ":memory:" for an in-memory database.
            Parent directory is created automatically if it doesn't exist.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def open(self) -> None:
        """Open the connection and apply schema DDL."""
        if self._conn is not None:
            return

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row

        # WAL mode: readers don't block writers and writers don't block readers
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        self._apply_schema()
        self._prune_old_sessions()

        logger.info("MemoryDB opened | path=%s", self._db_path)

    def close(self) -> None:
        """Close the connection cleanly."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("MemoryDB closed | path=%s", self._db_path)

    def __enter__(self) -> "MemoryDB":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Raw connection for read queries. Use transaction() for writes."""
        if self._conn is None:
            raise RuntimeError("MemoryDB is not open. Call open() first.")
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for transactional writes.
        Commits on success, rolls back on exception.
        Serialised with an internal lock.

        Usage
        -----
        with db.transaction() as conn:
            conn.execute("INSERT ...")
            conn.execute("UPDATE ...")
        """
        if self._conn is None:
            raise RuntimeError("MemoryDB is not open. Call open() first.")

        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Convenience read-only execute (no transaction)."""
        return self.conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    # -----------------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Run all DDL statements. Idempotent — safe to call on every open."""
        with self.transaction() as conn:
            for ddl in ALL_DDL:
                conn.execute(ddl)
        logger.debug("Schema applied successfully")

    # -----------------------------------------------------------------------
    # Garbage collection
    # -----------------------------------------------------------------------

    def _prune_old_sessions(self, keep_days: int = 30) -> None:
        """
        Delete sessions (and their episodes) older than keep_days on startup.
        Entities and semantic_facts are kept — they represent long-term knowledge.
        """
        if self._db_path == ":memory:":
            return   # nothing to prune in tests

        import time
        cutoff = time.time() - (keep_days * 86400)

        rows = self.fetchall(
            "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
            (cutoff,),
        )
        if not rows:
            return

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))

        with self.transaction() as conn:
            conn.execute(
                f"DELETE FROM episodes WHERE session_id IN ({placeholders})", ids
            )
            conn.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})", ids
            )
            conn.execute("VACUUM")

        logger.info(
            "Pruned %d sessions older than %d days", len(ids), keep_days
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def stats(self) -> dict:
        """Row counts for all tables. Useful for logs and the UI."""
        tables = ("sessions", "episodes", "entities", "relationships", "semantic_facts")
        result = {}
        for table in tables:
            row = self.fetchone(f"SELECT COUNT(*) as n FROM {table}")
            result[table] = row["n"] if row else 0
        return result
