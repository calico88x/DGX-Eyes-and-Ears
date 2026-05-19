"""
app/memory/writer.py

Write operations for the memory system.

MemoryWriter handles all inserts and updates. It accepts an optional
embed_fn for generating embeddings at write time. If embed_fn is None,
records are written without embeddings (still retrievable by recency).

Embeddings are numpy float32 arrays serialised to bytes for SQLite storage.
"""

import json
import logging
import time
import uuid
from typing import Callable, Optional

import numpy as np

from app.memory.db import MemoryDB
from app.memory.schema import (
    CHARS_PER_TOKEN,
    ENTITY_TYPES,
    EPISODE_SOURCES,
    FACT_SOURCES,
    ROLES,
)

logger = logging.getLogger(__name__)

# Type alias for an embedding function
EmbedFn = Callable[[str], Optional[list[float]]]


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> float:
    return time.time()


def _encode(vector: list[float]) -> bytes:
    """Serialise float list → bytes for SQLite BLOB storage."""
    return np.array(vector, dtype=np.float32).tobytes()


def _decode(blob: bytes) -> np.ndarray:
    """Deserialise BLOB → float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class MemoryWriter:
    """
    Write interface for the Eyes and Ears memory system.

    All methods are safe to call without embeddings — records will be stored
    and retrievable by recency/salience even if semantic search is unavailable.

    Parameters
    ----------
    db : MemoryDB
        Open database instance.
    embed_fn : EmbedFn | None
        Callable that takes a string and returns a float list (embedding).
        Pass None to skip embedding generation (recency-only retrieval).
    """

    def __init__(self, db: MemoryDB, embed_fn: Optional[EmbedFn] = None):
        self._db       = db
        self._embed_fn = embed_fn

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------

    def create_session(self, tags: Optional[list[str]] = None) -> str:
        """
        Open a new session. Returns the session ID.
        Call at the start of each conversation loop.
        """
        session_id = _new_id()
        now        = _now()

        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at, tags) VALUES (?, ?, ?)",
                (session_id, now, json.dumps(tags or []))
            )

        logger.info("Session created | id=%s", session_id)
        return session_id

    def close_session(self, session_id: str, summary: str = "") -> None:
        """
        Mark a session as ended and write its summary + embedding.
        Call when the conversation loop stops.
        """
        now       = _now()
        embedding = self._embed(summary) if summary else None

        with self._db.transaction() as conn:
            conn.execute(
                """UPDATE sessions
                      SET ended_at  = ?,
                          summary   = ?,
                          embedding = ?
                    WHERE id = ?""",
                (now, summary, embedding, session_id)
            )

        logger.info("Session closed | id=%s summary_len=%d", session_id, len(summary))

    # -----------------------------------------------------------------------
    # Episodes
    # -----------------------------------------------------------------------

    def write_episode(
        self,
        session_id:      str,
        role:            str,
        content:         str,
        source:          str          = "speech",
        salience_weight: float        = 0.5,
    ) -> str:
        """
        Store one exchange or event in the episode log.

        Parameters
        ----------
        session_id      : str    Active session ID from create_session().
        role            : str    One of ROLES ('user', 'assistant', 'system', 'event').
        content         : str    The actual text content of the episode.
        source          : str    Input origin — one of EPISODE_SOURCES.
        salience_weight : float  0.0–1.0. Higher = more likely to be recalled.

        Returns episode ID.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if source not in EPISODE_SOURCES:
            raise ValueError(f"source must be one of {EPISODE_SOURCES}, got {source!r}")

        episode_id = _new_id()
        now        = _now()
        embedding  = self._embed(content)

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO episodes
                       (id, session_id, role, source, content, embedding,
                        salience_weight, created_at, last_accessed, access_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (episode_id, session_id, role, source, content,
                 embedding, salience_weight, now, now)
            )

        logger.debug(
            "Episode written | session=%s role=%s source=%s len=%d",
            session_id, role, source, len(content)
        )
        return episode_id

    def update_episode_access(self, episode_id: str) -> None:
        """
        Record that an episode was retrieved and used.
        Frequently accessed memories score higher in retrieval.
        """
        now = _now()
        with self._db.transaction() as conn:
            conn.execute(
                """UPDATE episodes
                      SET last_accessed = ?,
                          access_count  = access_count + 1
                    WHERE id = ?""",
                (now, episode_id)
            )

    # -----------------------------------------------------------------------
    # Entities
    # -----------------------------------------------------------------------

    def upsert_entity(
        self,
        name:        str,
        entity_type: str   = "other",
        description: str   = "",
        importance:  float = 0.5,
    ) -> str:
        """
        Insert or update an entity. Returns the entity ID.

        If an entity with this name already exists, updates its description,
        last_seen timestamp, and importance if the new value is higher.
        """
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of {ENTITY_TYPES}, got {entity_type!r}")

        now       = _now()
        embedding = self._embed(f"{name}: {description}") if description else self._embed(name)

        existing = self._db.fetchone(
            "SELECT id, importance FROM entities WHERE name = ?", (name,)
        )

        if existing:
            entity_id      = existing["id"]
            new_importance = max(existing["importance"], importance)
            with self._db.transaction() as conn:
                conn.execute(
                    """UPDATE entities
                          SET description = ?,
                              embedding   = ?,
                              last_seen   = ?,
                              importance  = ?
                        WHERE id = ?""",
                    (description, embedding, now, new_importance, entity_id)
                )
            logger.debug("Entity updated | name=%r id=%s", name, entity_id)
        else:
            entity_id = _new_id()
            with self._db.transaction() as conn:
                conn.execute(
                    """INSERT INTO entities
                           (id, name, type, description, embedding,
                            first_seen, last_seen, importance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (entity_id, name, entity_type, description,
                     embedding, now, now, importance)
                )
            logger.debug("Entity created | name=%r id=%s", name, entity_id)

        return entity_id

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------

    def write_relationship(
        self,
        entity_a:     str,
        entity_b:     str,
        relationship: str,
        strength:     float = 0.5,
    ) -> str:
        """
        Write a relationship between two entity IDs.
        If an identical (a, b, relationship) triple exists, updates strength.
        Returns the relationship ID.
        """
        now = _now()

        existing = self._db.fetchone(
            """SELECT id FROM relationships
                WHERE entity_a = ? AND entity_b = ? AND relationship = ?""",
            (entity_a, entity_b, relationship)
        )

        if existing:
            rel_id = existing["id"]
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE relationships SET strength = ? WHERE id = ?",
                    (strength, rel_id)
                )
        else:
            rel_id = _new_id()
            with self._db.transaction() as conn:
                conn.execute(
                    """INSERT INTO relationships
                           (id, entity_a, entity_b, relationship, strength, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (rel_id, entity_a, entity_b, relationship, strength, now)
                )

        return rel_id

    # -----------------------------------------------------------------------
    # Semantic facts
    # -----------------------------------------------------------------------

    def write_fact(
        self,
        fact:              str,
        source_episode_id: Optional[str] = None,
        confidence:        float         = 1.0,
        source:            str           = "derived",
    ) -> str:
        """
        Store a persistent semantic fact.

        Facts are AI-extracted knowledge that should persist across sessions
        (e.g. "User prefers Python over JavaScript", "User's cat is named Max").

        Parameters
        ----------
        fact              : str   The fact as a plain sentence.
        source_episode_id : str   Optional episode this was derived from.
        confidence        : float 0.0–1.0. Lower for inferred/uncertain facts.
        source            : str   One of FACT_SOURCES.

        Returns fact ID.
        """
        if source not in FACT_SOURCES:
            raise ValueError(f"source must be one of {FACT_SOURCES}, got {source!r}")

        fact_id   = _new_id()
        now       = _now()
        embedding = self._embed(fact)

        with self._db.transaction() as conn:
            conn.execute(
                """INSERT INTO semantic_facts
                       (id, fact, source_episode_id, embedding,
                        confidence, created_at, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (fact_id, fact, source_episode_id, embedding,
                 confidence, now, source)
            )

        logger.debug("Fact written | source=%s len=%d", source, len(fact))
        return fact_id

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[bytes]:
        """Call embed_fn and serialise result. Returns None if no embed_fn."""
        if self._embed_fn is None or not text.strip():
            return None
        try:
            vector = self._embed_fn(text)
            return _encode(vector) if vector else None
        except Exception as exc:
            logger.warning("Embedding failed for text (len=%d): %s", len(text), exc)
            return None
