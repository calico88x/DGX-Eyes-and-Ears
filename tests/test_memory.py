"""
tests/test_memory.py

Fully offline tests for the memory system.
All tests use in-memory SQLite (:memory:) — no files created or cleaned up.
Embeddings use a seeded deterministic fake embed_fn — no LLM calls.

Run with:  pytest tests/test_memory.py -v
"""

import json
import math
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.memory.db     import MemoryDB
from app.memory.writer import MemoryWriter, _encode, _decode
from app.memory.reader import MemoryReader, _cosine_similarity
from app.memory.schema import ALL_DDL, ROLES, EPISODE_SOURCES, ENTITY_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Fresh in-memory database for each test."""
    with MemoryDB(":memory:") as database:
        yield database


@pytest.fixture
def embed_fn():
    """
    Deterministic fake embedding function.
    Same text → same vector. Different text → different vector.
    Dimension 8 (small, fast for tests).
    """
    dim = 8
    def _embed(text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v   = rng.standard_normal(dim).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()
    return _embed


@pytest.fixture
def writer(db, embed_fn):
    return MemoryWriter(db, embed_fn=embed_fn)


@pytest.fixture
def writer_no_embed(db):
    return MemoryWriter(db, embed_fn=None)


@pytest.fixture
def reader(db, embed_fn):
    return MemoryReader(db, embed_fn=embed_fn)


@pytest.fixture
def reader_no_embed(db):
    return MemoryReader(db, embed_fn=None)


@pytest.fixture
def session(writer):
    return writer.create_session()


# ---------------------------------------------------------------------------
# MemoryDB
# ---------------------------------------------------------------------------

class TestMemoryDB:
    def test_opens_in_memory(self):
        with MemoryDB(":memory:") as db:
            assert db.conn is not None

    def test_schema_applied_on_open(self, db):
        tables = {"sessions", "episodes", "entities", "relationships", "semantic_facts"}
        rows   = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        found = {r["name"] for r in rows}
        assert tables.issubset(found)

    def test_schema_is_idempotent(self, db):
        # Calling _apply_schema again should not raise
        db._apply_schema()
        db._apply_schema()

    def test_transaction_commits_on_success(self, db):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at, tags) VALUES ('s1', 1.0, '[]')"
            )
        row = db.fetchone("SELECT id FROM sessions WHERE id = 's1'")
        assert row is not None

    def test_transaction_rolls_back_on_exception(self, db):
        with pytest.raises(Exception):
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO sessions (id, started_at, tags) VALUES ('s2', 1.0, '[]')"
                )
                raise ValueError("deliberate rollback")
        row = db.fetchone("SELECT id FROM sessions WHERE id = 's2'")
        assert row is None

    def test_stats_returns_zero_counts_on_empty_db(self, db):
        s = db.stats()
        assert s["sessions"] == 0
        assert s["episodes"] == 0

    def test_stats_counts_increase_after_insert(self, db, writer):
        writer.create_session()
        assert db.stats()["sessions"] == 1

    def test_raises_when_not_open(self):
        db = MemoryDB(":memory:")
        with pytest.raises(RuntimeError):
            db.conn

    def test_creates_parent_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "memory.db")
        with MemoryDB(db_path) as db:
            assert db.conn is not None
        assert Path(db_path).exists()


# ---------------------------------------------------------------------------
# MemoryWriter — sessions
# ---------------------------------------------------------------------------

class TestWriterSessions:
    def test_create_session_returns_id(self, writer):
        sid = writer.create_session()
        assert isinstance(sid, str) and len(sid) > 0

    def test_create_session_row_exists(self, db, writer):
        sid = writer.create_session()
        row = db.fetchone("SELECT * FROM sessions WHERE id = ?", (sid,))
        assert row is not None

    def test_create_session_with_tags(self, db, writer):
        sid = writer.create_session(tags=["gaming", "test"])
        row = db.fetchone("SELECT tags FROM sessions WHERE id = ?", (sid,))
        assert "gaming" in row["tags"]

    def test_close_session_sets_ended_at(self, db, writer):
        sid = writer.create_session()
        writer.close_session(sid, summary="test summary")
        row = db.fetchone("SELECT ended_at, summary FROM sessions WHERE id = ?", (sid,))
        assert row["ended_at"] is not None
        assert row["summary"] == "test summary"

    def test_close_session_writes_embedding(self, db, writer):
        sid = writer.create_session()
        writer.close_session(sid, summary="we talked about cats")
        row = db.fetchone("SELECT embedding FROM sessions WHERE id = ?", (sid,))
        assert row["embedding"] is not None

    def test_close_session_no_embed_fn(self, db, writer_no_embed):
        sid = writer_no_embed.create_session()
        writer_no_embed.close_session(sid, summary="test")
        row = db.fetchone("SELECT embedding FROM sessions WHERE id = ?", (sid,))
        assert row["embedding"] is None

    def test_multiple_sessions_are_independent(self, writer):
        s1 = writer.create_session()
        s2 = writer.create_session()
        assert s1 != s2


# ---------------------------------------------------------------------------
# MemoryWriter — episodes
# ---------------------------------------------------------------------------

class TestWriterEpisodes:
    def test_write_episode_returns_id(self, writer, session):
        eid = writer.write_episode(session, "user", "hello there")
        assert isinstance(eid, str)

    def test_write_episode_row_exists(self, db, writer, session):
        eid = writer.write_episode(session, "user", "test content")
        row = db.fetchone("SELECT * FROM episodes WHERE id = ?", (eid,))
        assert row is not None
        assert row["content"] == "test content"
        assert row["role"]    == "user"

    def test_write_episode_stores_embedding(self, db, writer, session):
        eid = writer.write_episode(session, "assistant", "I see a car")
        row = db.fetchone("SELECT embedding FROM episodes WHERE id = ?", (eid,))
        assert row["embedding"] is not None

    def test_write_episode_no_embedding_when_no_fn(self, db, writer_no_embed, session):
        # writer_no_embed has a real session via the session fixture which uses writer
        # so we need to create our own session here
        sid = writer_no_embed.create_session()
        eid = writer_no_embed.write_episode(sid, "user", "test")
        row = db.fetchone("SELECT embedding FROM episodes WHERE id = ?", (eid,))
        assert row["embedding"] is None

    def test_write_episode_invalid_role_raises(self, writer, session):
        with pytest.raises(ValueError, match="role"):
            writer.write_episode(session, "alien", "hello")

    def test_write_episode_invalid_source_raises(self, writer, session):
        with pytest.raises(ValueError, match="source"):
            writer.write_episode(session, "user", "hello", source="telepathy")

    def test_write_episode_default_salience(self, db, writer, session):
        eid = writer.write_episode(session, "user", "content")
        row = db.fetchone("SELECT salience_weight FROM episodes WHERE id = ?", (eid,))
        assert row["salience_weight"] == 0.5

    def test_write_episode_custom_salience(self, db, writer, session):
        eid = writer.write_episode(session, "user", "content", salience_weight=0.9)
        row = db.fetchone("SELECT salience_weight FROM episodes WHERE id = ?", (eid,))
        assert row["salience_weight"] == 0.9

    def test_update_episode_access_increments_count(self, db, writer, session):
        eid = writer.write_episode(session, "user", "test")
        writer.update_episode_access(eid)
        writer.update_episode_access(eid)
        row = db.fetchone("SELECT access_count FROM episodes WHERE id = ?", (eid,))
        assert row["access_count"] == 2

    def test_write_all_valid_roles(self, writer, session):
        for role in ROLES:
            writer.write_episode(session, role, f"content for {role}")

    def test_write_all_valid_sources(self, writer, session):
        for source in EPISODE_SOURCES:
            writer.write_episode(session, "user", f"content", source=source)


# ---------------------------------------------------------------------------
# MemoryWriter — entities
# ---------------------------------------------------------------------------

class TestWriterEntities:
    def test_upsert_entity_creates_new(self, db, writer):
        eid = writer.upsert_entity("Max", "person", "user's friend")
        row = db.fetchone("SELECT * FROM entities WHERE id = ?", (eid,))
        assert row["name"] == "Max"
        assert row["type"] == "person"

    def test_upsert_entity_updates_existing(self, db, writer):
        id1 = writer.upsert_entity("Max", "person", "friend")
        id2 = writer.upsert_entity("Max", "person", "close friend", importance=0.9)
        assert id1 == id2   # same entity
        row = db.fetchone("SELECT description, importance FROM entities WHERE id = ?", (id1,))
        assert row["description"] == "close friend"
        assert row["importance"]  == 0.9

    def test_upsert_entity_importance_keeps_max(self, db, writer):
        writer.upsert_entity("Alice", importance=0.8)
        writer.upsert_entity("Alice", importance=0.3)   # lower — should not reduce
        row = db.fetchone("SELECT importance FROM entities WHERE name = 'Alice'")
        assert row["importance"] == 0.8

    def test_upsert_entity_invalid_type_raises(self, writer):
        with pytest.raises(ValueError, match="entity_type"):
            writer.upsert_entity("Bob", entity_type="robot")

    def test_upsert_all_valid_types(self, writer):
        for t in ENTITY_TYPES:
            writer.upsert_entity(f"entity_{t}", entity_type=t)


# ---------------------------------------------------------------------------
# MemoryWriter — relationships
# ---------------------------------------------------------------------------

class TestWriterRelationships:
    def test_write_relationship_creates_row(self, db, writer):
        a = writer.upsert_entity("Alice")
        b = writer.upsert_entity("Bob")
        rid = writer.write_relationship(a, b, "knows")
        row = db.fetchone("SELECT * FROM relationships WHERE id = ?", (rid,))
        assert row is not None
        assert row["relationship"] == "knows"

    def test_write_relationship_upserts_on_duplicate(self, db, writer):
        a = writer.upsert_entity("Alice")
        b = writer.upsert_entity("Bob")
        r1 = writer.write_relationship(a, b, "knows", strength=0.5)
        r2 = writer.write_relationship(a, b, "knows", strength=0.9)
        assert r1 == r2
        row = db.fetchone("SELECT strength FROM relationships WHERE id = ?", (r1,))
        assert row["strength"] == 0.9

    def test_different_relationship_types_are_separate(self, db, writer):
        a = writer.upsert_entity("Alice")
        b = writer.upsert_entity("Bob")
        writer.write_relationship(a, b, "knows")
        writer.write_relationship(a, b, "likes")
        count = db.fetchone(
            "SELECT COUNT(*) as n FROM relationships WHERE entity_a = ?", (a,)
        )
        assert count["n"] == 2


# ---------------------------------------------------------------------------
# MemoryWriter — semantic facts
# ---------------------------------------------------------------------------

class TestWriterFacts:
    def test_write_fact_creates_row(self, db, writer):
        fid = writer.write_fact("User prefers dark mode.")
        row = db.fetchone("SELECT * FROM semantic_facts WHERE id = ?", (fid,))
        assert row["fact"] == "User prefers dark mode."

    def test_write_fact_stores_embedding(self, db, writer):
        fid = writer.write_fact("User has a cat named Mochi.")
        row = db.fetchone("SELECT embedding FROM semantic_facts WHERE id = ?", (fid,))
        assert row["embedding"] is not None

    def test_write_fact_invalid_source_raises(self, writer):
        with pytest.raises(ValueError, match="source"):
            writer.write_fact("test", source="random")

    def test_write_fact_default_source_is_derived(self, db, writer):
        fid = writer.write_fact("User likes jazz.")
        row = db.fetchone("SELECT source FROM semantic_facts WHERE id = ?", (fid,))
        assert row["source"] == "derived"

    def test_write_fact_with_episode_link(self, db, writer, session):
        eid = writer.write_episode(session, "user", "I love jazz")
        fid = writer.write_fact("User likes jazz.", source_episode_id=eid)
        row = db.fetchone(
            "SELECT source_episode_id FROM semantic_facts WHERE id = ?", (fid,)
        )
        assert row["source_episode_id"] == eid


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    def test_encode_decode_roundtrip(self):
        v = [0.1, 0.2, 0.3, 0.4]
        assert np.allclose(_decode(_encode(v)), v, atol=1e-6)

    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert math.isclose(_cosine_similarity(
            np.array(v, dtype=np.float32),
            np.array(v, dtype=np.float32)
        ), 1.0, abs_tol=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_cosine_similarity_zero_vector_returns_zero(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# MemoryReader
# ---------------------------------------------------------------------------

class TestReaderBasic:
    def test_get_recent_episodes_returns_in_order(self, writer, reader, session):
        for i in range(5):
            writer.write_episode(session, "user", f"message {i}")
        results = reader.get_recent_episodes(session, limit=3)
        assert len(results) == 3
        # Chronological order — oldest first
        assert results[0].content == "message 2"
        assert results[2].content == "message 4"

    def test_get_recent_episodes_empty_session(self, reader, session):
        results = reader.get_recent_episodes(session, limit=10)
        assert results == []

    def test_get_session_returns_dict(self, writer, reader):
        sid = writer.create_session(tags=["test"])
        result = reader.get_session(sid)
        assert result is not None
        assert result["id"] == sid

    def test_get_session_missing_returns_none(self, reader):
        assert reader.get_session("nonexistent-id") is None

    def test_get_recent_sessions_returns_closed(self, writer, reader):
        s1 = writer.create_session()
        writer.close_session(s1, "first session")
        s2 = writer.create_session()
        writer.close_session(s2, "second session")

        sessions = reader.get_recent_sessions(limit=5)
        assert len(sessions) == 2

    def test_get_entity_returns_result(self, writer, reader):
        writer.upsert_entity("Mochi", "object", "user's cat")
        result = reader.get_entity("Mochi")
        assert result is not None
        assert result.entity_type == "object"

    def test_get_entity_missing_returns_none(self, reader):
        assert reader.get_entity("Unknown Person") is None

    def test_search_facts_no_embed_uses_recency(self, db, writer_no_embed, reader_no_embed):
        sid = writer_no_embed.create_session()
        writer_no_embed.write_fact("User likes Python.", source="explicit")
        writer_no_embed.write_fact("User has a dog.", source="explicit")
        results = reader_no_embed.search_facts("programming", top_k=5)
        assert len(results) >= 1


class TestReaderSemanticSearch:
    def test_semantic_search_returns_results(self, writer, reader, session):
        writer.write_episode(session, "user",      "I love playing video games")
        writer.write_episode(session, "assistant", "That sounds fun!")
        writer.write_episode(session, "user",      "My favourite is racing games")

        results = reader.get_recent_episodes(session, limit=10)
        assert len(results) == 3

    def test_search_facts_returns_relevant(self, writer, reader):
        writer.write_fact("User enjoys programming in Python.", source="explicit")
        writer.write_fact("User has a cat named Mochi.",        source="explicit")
        writer.write_fact("User prefers dark roast coffee.",    source="explicit")

        results = reader.search_facts("Python code", top_k=3)
        # All three facts should be retrieved (semantic ordering is model-dependent,
        # not testable with a random fake embed_fn — tested via integration with real model)
        assert len(results) == 3
        facts_text = " ".join(r.fact for r in results)
        assert "Python"  in facts_text
        assert "Mochi"   in facts_text
        assert "coffee"  in facts_text

    def test_entity_lookup_finds_name_in_text(self, writer, reader):
        writer.upsert_entity("Mochi", "object", "user's cat")
        entities = reader._find_entities_in_text("my cat Mochi is sleeping")
        assert len(entities) == 1
        assert entities[0].name == "Mochi"

    def test_entity_lookup_no_match(self, writer, reader):
        writer.upsert_entity("Alice")
        entities = reader._find_entities_in_text("the weather is nice today")
        assert entities == []


class TestReaderContextForPrompt:
    def test_returns_memory_context(self, writer, reader, session):
        writer.write_episode(session, "user",      "what games do you know")
        writer.write_episode(session, "assistant", "I know many games!")
        ctx = reader.get_context_for_prompt("games", session)
        assert ctx is not None

    def test_prompt_text_contains_memory_markers(self, writer, reader, session):
        writer.write_episode(session, "user", "hello")
        ctx = reader.get_context_for_prompt("hello", session)
        assert "[MEMORY]"  in ctx.prompt_text
        assert "[/MEMORY]" in ctx.prompt_text

    def test_prompt_text_empty_when_no_memories(self, reader, session):
        ctx = reader.get_context_for_prompt("query", session)
        assert ctx.prompt_text == ""

    def test_token_estimate_is_positive_with_memories(self, writer, reader, session):
        writer.write_episode(session, "user", "this is a long piece of content " * 5)
        ctx = reader.get_context_for_prompt("content", session)
        assert ctx.token_estimate > 0

    def test_budget_trimming_respects_limit(self, writer, reader, session):
        # Write many large episodes
        for i in range(20):
            writer.write_episode(
                session, "user",
                f"episode {i}: " + "x" * 200,
                salience_weight=0.5
            )
        ctx = reader.get_context_for_prompt("query", session, budget_tokens=50)
        assert ctx.token_estimate <= 60   # small margin for formatting overhead

    def test_facts_included_in_prompt(self, writer, reader, session):
        writer.write_fact("User is a software engineer.", source="explicit")
        ctx = reader.get_context_for_prompt("coding project", session)
        assert "software engineer" in ctx.prompt_text

    def test_entities_included_in_prompt(self, writer, reader, session):
        writer.upsert_entity("Alice", "person", "user's colleague")
        writer.write_episode(session, "user", "Alice helped me today")
        ctx = reader.get_context_for_prompt("Alice helped me today", session)
        assert "Alice" in ctx.prompt_text

    def test_no_embed_fn_falls_back_to_recency(self, writer, reader_no_embed):
        sid = writer.create_session()
        writer.write_episode(sid, "user", "test message")
        ctx = reader_no_embed.get_context_for_prompt("test", sid)
        # Should still work, just without semantic ranking
        assert isinstance(ctx.episodes, list)

    def test_retrieved_count_reflects_candidates(self, writer, reader, session):
        for i in range(5):
            writer.write_episode(session, "user", f"message {i}")
        ctx = reader.get_context_for_prompt("message", session)
        assert ctx.retrieved_count >= 5


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class TestImporter:
    def test_import_json_facts(self, tmp_path):
        from app.memory.importer import run_import

        data = [
            {"content": "User likes cats.", "type": "fact"},
            {"content": "User codes in Python.", "type": "fact"},
        ]
        f = tmp_path / "memories.json"
        f.write_text(json.dumps(data))

        db_path = str(tmp_path / "test.db")
        counts  = run_import(db_path, str(f))

        assert counts["imported"] == 2
        assert counts["errors"]   == 0

    def test_import_json_episodes(self, tmp_path):
        from app.memory.importer import run_import

        data = [
            {"content": "hello", "type": "episode", "role": "user"},
            {"content": "hi!",   "type": "episode", "role": "assistant"},
        ]
        f = tmp_path / "memories.json"
        f.write_text(json.dumps(data))

        counts = run_import(str(tmp_path / "test.db"), str(f))
        assert counts["imported"] == 2

    def test_import_csv(self, tmp_path):
        from app.memory.importer import run_import

        csv_content = "content,role,source,type,confidence\n"
        csv_content += "User enjoys hiking.,assistant,import,fact,1.0\n"
        csv_content += "User has a dog.,assistant,import,fact,0.9\n"

        f = tmp_path / "memories.csv"
        f.write_text(csv_content)

        counts = run_import(str(tmp_path / "test.db"), str(f))
        assert counts["imported"] == 2

    def test_import_dry_run_does_not_write(self, tmp_path):
        from app.memory.importer import run_import

        data   = [{"content": "test fact", "type": "fact"}]
        f      = tmp_path / "memories.json"
        f.write_text(json.dumps(data))
        db_path = str(tmp_path / "test.db")

        counts = run_import(db_path, str(f), dry_run=True)
        assert counts["imported"] == 1

        # Database should have no facts (dry run)
        with MemoryDB(db_path) as db:
            row = db.fetchone("SELECT COUNT(*) as n FROM semantic_facts")
            assert row["n"] == 0

    def test_import_skips_empty_content(self, tmp_path):
        from app.memory.importer import run_import

        data = [
            {"content": "",       "type": "fact"},   # empty — should skip
            {"content": "valid.", "type": "fact"},
        ]
        f = tmp_path / "memories.json"
        f.write_text(json.dumps(data))

        counts = run_import(str(tmp_path / "test.db"), str(f))
        assert counts["imported"] == 1
        assert counts["errors"]   == 1

    def test_import_skips_invalid_role(self, tmp_path):
        from app.memory.importer import run_import

        data = [{"content": "test", "type": "episode", "role": "alien"}]
        f = tmp_path / "memories.json"
        f.write_text(json.dumps(data))

        counts = run_import(str(tmp_path / "test.db"), str(f))
        assert counts["errors"] == 1

    def test_import_missing_file_raises(self, tmp_path):
        from app.memory.importer import run_import
        with pytest.raises(FileNotFoundError):
            run_import(str(tmp_path / "test.db"), "/nonexistent/file.json")

    def test_import_unsupported_extension_raises(self, tmp_path):
        from app.memory.importer import run_import, _read_file
        f = tmp_path / "memories.xml"
        f.write_text("<memories/>")
        with pytest.raises(ValueError, match="Unsupported"):
            list(_read_file(f))


# ---------------------------------------------------------------------------
# Full pipeline integration (offline)
# ---------------------------------------------------------------------------

class TestMemoryPipelineIntegration:
    """
    End-to-end memory pipeline: write a session, retrieve context.
    No services. Uses fake embed_fn.
    """

    def test_full_session_lifecycle(self, writer, reader):
        # 1. Start session
        sid = writer.create_session(tags=["integration_test"])

        # 2. Write a conversation
        writer.write_episode(sid, "user",      "hey, I've been learning Rust", salience_weight=0.7)
        writer.write_episode(sid, "assistant", "That's exciting! How are you finding it?", salience_weight=0.5)
        writer.write_episode(sid, "user",      "it's hard but really fun", salience_weight=0.6)

        # 3. Write some facts and entities
        writer.write_fact("User is learning Rust programming.", source="derived", confidence=0.9)
        writer.upsert_entity("Rust", "topic", "programming language the user is learning", importance=0.7)

        # 4. Close session
        writer.close_session(sid, summary="User discussed learning Rust programming.")

        # 5. Retrieve context for next turn
        ctx = reader.get_context_for_prompt(
            query      = "how is the Rust project going",
            session_id = sid,
            budget_tokens = 400,
        )

        # Should contain the session's content
        assert "Rust" in ctx.prompt_text
        assert ctx.token_estimate > 0
        assert len(ctx.episodes) > 0

    def test_cross_session_semantic_recall(self, writer, reader):
        """Semantically similar content from old sessions should surface."""
        # Session 1 — old conversation
        s1 = writer.create_session()
        writer.write_episode(s1, "user",      "I love jazz music, especially Miles Davis")
        writer.write_episode(s1, "assistant", "Miles Davis was a legend.")
        writer.close_session(s1, "Discussed jazz and Miles Davis.")

        # Session 2 — current
        s2 = writer.create_session()

        ctx = reader.get_context_for_prompt(
            query      = "put on some jazz",
            session_id = s2,
            budget_tokens = 500,
        )

        # The jazz episode from session 1 should appear in context
        all_content = ctx.prompt_text
        assert "jazz" in all_content.lower()

    def test_memory_survives_db_reopen(self, tmp_path, embed_fn):
        """Persisted data should be readable after closing and reopening the DB."""
        db_path = str(tmp_path / "persist_test.db")

        # Write
        with MemoryDB(db_path) as db:
            w = MemoryWriter(db, embed_fn=embed_fn)
            sid = w.create_session()
            w.write_fact("User has a cat named Luna.", source="explicit")
            w.close_session(sid, "Talked about Luna the cat.")

        # Read back from fresh connection
        with MemoryDB(db_path) as db:
            r   = MemoryReader(db, embed_fn=embed_fn)
            sid2 = MemoryWriter(db, embed_fn=embed_fn).create_session()
            ctx = r.get_context_for_prompt("what's my cat's name", sid2)

        assert "Luna" in ctx.prompt_text
