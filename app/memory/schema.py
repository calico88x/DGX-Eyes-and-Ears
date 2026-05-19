"""
app/memory/schema.py

SQLite schema for the Eyes and Ears memory system.

Five tables model human-like memory:

  sessions       — each conversation is a bounded session
  episodes       — individual exchanges/events within a session
  entities       — people, places, topics the AI knows about
  relationships  — connections between entities
  semantic_facts — persistent facts independent of any episode

Embeddings are stored as BLOBs (numpy float32 arrays → bytes).
Timestamps are stored as REAL (Unix epoch, seconds).

All DDL is idempotent (IF NOT EXISTS) — safe to call on every startup.
"""

# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT    PRIMARY KEY,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    summary     TEXT    DEFAULT '',
    embedding   BLOB,
    tags        TEXT    DEFAULT '[]'   -- JSON array of strings
);
"""

CREATE_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT    PRIMARY KEY,
    session_id      TEXT    NOT NULL REFERENCES sessions(id),
    role            TEXT    NOT NULL,   -- 'user' | 'assistant' | 'system' | 'event'
    source          TEXT    NOT NULL,   -- 'speech' | 'vision' | 'system' | 'import'
    content         TEXT    NOT NULL,
    embedding       BLOB,
    salience_weight REAL    NOT NULL DEFAULT 0.5,
    created_at      REAL    NOT NULL,
    last_accessed   REAL    NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0
);
"""

CREATE_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    type        TEXT    NOT NULL DEFAULT 'other',  -- 'person'|'place'|'topic'|'object'|'other'
    description TEXT    DEFAULT '',
    embedding   BLOB,
    first_seen  REAL    NOT NULL,
    last_seen   REAL    NOT NULL,
    importance  REAL    NOT NULL DEFAULT 0.5
);
"""

CREATE_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS relationships (
    id           TEXT    PRIMARY KEY,
    entity_a     TEXT    NOT NULL REFERENCES entities(id),
    entity_b     TEXT    NOT NULL REFERENCES entities(id),
    relationship TEXT    NOT NULL,   -- e.g. 'knows', 'likes', 'related_to'
    strength     REAL    NOT NULL DEFAULT 0.5,
    created_at   REAL    NOT NULL,
    UNIQUE(entity_a, entity_b, relationship)
);
"""

CREATE_SEMANTIC_FACTS = """
CREATE TABLE IF NOT EXISTS semantic_facts (
    id                TEXT    PRIMARY KEY,
    fact              TEXT    NOT NULL,
    source_episode_id TEXT    REFERENCES episodes(id),
    embedding         BLOB,
    confidence        REAL    NOT NULL DEFAULT 1.0,
    created_at        REAL    NOT NULL,
    source            TEXT    NOT NULL DEFAULT 'derived'  -- 'derived'|'import'|'explicit'
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_episodes_session    ON episodes(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_created    ON episodes(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_salience   ON episodes(salience_weight DESC);",
    "CREATE INDEX IF NOT EXISTS idx_entities_name       ON entities(name);",
    "CREATE INDEX IF NOT EXISTS idx_facts_created       ON semantic_facts(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_relationships_a     ON relationships(entity_a);",
    "CREATE INDEX IF NOT EXISTS idx_relationships_b     ON relationships(entity_b);",
]

# ---------------------------------------------------------------------------
# All DDL in execution order
# ---------------------------------------------------------------------------

ALL_DDL = [
    CREATE_SESSIONS,
    CREATE_EPISODES,
    CREATE_ENTITIES,
    CREATE_RELATIONSHIPS,
    CREATE_SEMANTIC_FACTS,
    *INDEXES,
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid values for episodes.role
ROLES = ("user", "assistant", "system", "event")

# Valid values for episodes.source / semantic_facts.source
EPISODE_SOURCES = ("speech", "vision", "audio_event", "system", "import")
FACT_SOURCES    = ("derived", "import", "explicit")

# Valid values for entities.type
ENTITY_TYPES = ("person", "place", "topic", "object", "other")

# Embedding vector dimension (must match the model in config.yaml)
EMBEDDING_DIM = 1536

# Approximate chars per token — used for context budget trimming
CHARS_PER_TOKEN = 4
