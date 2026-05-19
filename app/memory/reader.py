"""
app/memory/reader.py

Read and retrieval operations for the memory system.

The primary method is get_context_for_prompt() — it assembles a
ranked, budget-trimmed memory context string ready for injection
into the LLM system prompt every turn.

Retrieval scoring combines three signals:
    score = (SEMANTIC_W  * cosine_similarity)
          + (RECENCY_W   * recency_score)        # decays over 24h
          + (SALIENCE_W  * salience_boost)        # salience + access frequency

Weights are tunable via the constructor.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from app.memory.db import MemoryDB
from app.memory.schema import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Optional[list[float]]]


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    id:              str
    session_id:      str
    role:            str
    source:          str
    content:         str
    salience_weight: float
    created_at:      float
    access_count:    int
    score:           float = 0.0


@dataclass
class FactResult:
    id:         str
    fact:       str
    confidence: float
    source:     str
    created_at: float
    score:      float = 0.0


@dataclass
class EntityResult:
    id:          str
    name:        str
    entity_type: str
    description: str
    importance:  float


@dataclass
class MemoryContext:
    """
    Assembled memory context ready for LLM injection.
    """
    episodes:        list[EpisodeResult]
    facts:           list[FactResult]
    entities:        list[EntityResult]
    prompt_text:     str    # formatted string to inject into the system prompt
    token_estimate:  int
    retrieved_count: int    # total records considered before trimming


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class MemoryReader:
    """
    Read and retrieval interface for the memory system.

    Parameters
    ----------
    db : MemoryDB
        Open database instance.
    embed_fn : EmbedFn | None
        Embedding function. If None, semantic search is skipped and
        retrieval falls back to recency + salience only.
    semantic_weight : float
        Weight of cosine similarity in the composite score (default 0.5).
    recency_weight : float
        Weight of recency decay in the composite score (default 0.3).
    salience_weight : float
        Weight of salience/access boost in the composite score (default 0.2).
    recency_half_life_hours : float
        Hours after which recency score halves (default 24).
    """

    def __init__(
        self,
        db:                     MemoryDB,
        embed_fn:               Optional[EmbedFn] = None,
        semantic_weight:        float = 0.5,
        recency_weight:         float = 0.3,
        salience_weight:        float = 0.2,
        recency_half_life_hours: float = 24.0,
    ):
        self._db        = db
        self._embed_fn  = embed_fn
        self._sem_w     = semantic_weight
        self._rec_w     = recency_weight
        self._sal_w     = salience_weight
        self._half_life = recency_half_life_hours * 3600  # seconds

    # -----------------------------------------------------------------------
    # Primary retrieval
    # -----------------------------------------------------------------------

    def get_context_for_prompt(
        self,
        query:          str,
        session_id:     str,
        budget_tokens:  int = 800,
        recent_n:       int = 6,
        semantic_top_k: int = 8,
        fact_top_k:     int = 5,
    ) -> MemoryContext:
        """
        Assemble ranked memory context for one LLM turn.

        Strategy
        --------
        1. Always include the last `recent_n` episodes from the current session
           (working memory — what just happened).
        2. Semantic search across all episodes for `semantic_top_k` most
           relevant past memories.
        3. Retrieve `fact_top_k` relevant semantic facts.
        4. Look up entities mentioned in the query.
        5. Score, deduplicate, rank, and trim to budget_tokens.

        Parameters
        ----------
        query          : str  Current user speech or context description.
        session_id     : str  Active session — its recent episodes always included.
        budget_tokens  : int  Max tokens to consume in the prompt. Default 800.
        recent_n       : int  Recent episodes from current session to always include.
        semantic_top_k : int  Max semantically similar episodes to retrieve.
        fact_top_k     : int  Max facts to retrieve.

        Returns MemoryContext with formatted prompt_text ready for injection.
        """
        query_embedding = self._embed(query)
        now             = time.time()
        seen_ids        = set()
        all_episodes: list[EpisodeResult] = []

        # 1. Recent episodes from current session (always included)
        recent = self._get_recent_episodes(session_id, recent_n)
        for ep in recent:
            seen_ids.add(ep.id)
            ep.score = 1.0   # recent = maximum score
            all_episodes.append(ep)

        # 2. Semantic search across all episodes
        if query_embedding is not None:
            semantic = self._semantic_episode_search(
                query_embedding, semantic_top_k + recent_n, now
            )
            for ep in semantic:
                if ep.id not in seen_ids:
                    seen_ids.add(ep.id)
                    all_episodes.append(ep)

        # 3. Semantic facts
        facts: list[FactResult] = []
        if query_embedding is not None:
            facts = self._semantic_fact_search(query_embedding, fact_top_k, now)
        else:
            facts = self._recent_facts(fact_top_k)

        # 4. Entities mentioned in query
        entities = self._find_entities_in_text(query)

        # 5. Sort episodes by score, trim to budget
        all_episodes.sort(key=lambda e: e.score, reverse=True)
        total_retrieved = len(all_episodes) + len(facts)

        episodes_out, facts_out, token_estimate = self._trim_to_budget(
            all_episodes, facts, budget_tokens
        )

        prompt_text = self._format_context(episodes_out, facts_out, entities)

        return MemoryContext(
            episodes        = episodes_out,
            facts           = facts_out,
            entities        = entities,
            prompt_text     = prompt_text,
            token_estimate  = token_estimate,
            retrieved_count = total_retrieved,
        )

    # -----------------------------------------------------------------------
    # Individual query methods
    # -----------------------------------------------------------------------

    def get_recent_episodes(
        self,
        session_id: str,
        limit:      int = 10,
    ) -> list[EpisodeResult]:
        """Most recent N episodes from a session. No scoring."""
        return self._get_recent_episodes(session_id, limit)

    def get_session(self, session_id: str) -> Optional[dict]:
        """Fetch a single session record."""
        row = self._db.fetchone(
            "SELECT id, started_at, ended_at, summary, tags FROM sessions WHERE id = ?",
            (session_id,)
        )
        if row is None:
            return None
        return {
            "id":         row["id"],
            "started_at": row["started_at"],
            "ended_at":   row["ended_at"],
            "summary":    row["summary"],
            "tags":       row["tags"],
        }

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        """Most recent N closed sessions with summaries."""
        rows = self._db.fetchall(
            """SELECT id, started_at, ended_at, summary, tags
                 FROM sessions
                WHERE ended_at IS NOT NULL
             ORDER BY started_at DESC
                LIMIT ?""",
            (limit,)
        )
        return [dict(r) for r in rows]

    def get_entity(self, name: str) -> Optional[EntityResult]:
        """Exact name lookup for a single entity."""
        row = self._db.fetchone(
            "SELECT id, name, type, description, importance FROM entities WHERE name = ?",
            (name,)
        )
        if row is None:
            return None
        return EntityResult(
            id=row["id"], name=row["name"], entity_type=row["type"],
            description=row["description"], importance=row["importance"]
        )

    def search_facts(self, query: str, top_k: int = 5) -> list[FactResult]:
        """Semantic search over facts for a plain text query."""
        embedding = self._embed(query)
        if embedding is None:
            return self._recent_facts(top_k)
        return self._semantic_fact_search(np.array(embedding, dtype=np.float32), top_k, time.time())

    # -----------------------------------------------------------------------
    # Internal — queries
    # -----------------------------------------------------------------------

    def _get_recent_episodes(self, session_id: str, limit: int) -> list[EpisodeResult]:
        rows = self._db.fetchall(
            """SELECT id, session_id, role, source, content,
                      salience_weight, created_at, access_count
                 FROM episodes
                WHERE session_id = ?
             ORDER BY created_at DESC
                LIMIT ?""",
            (session_id, limit)
        )
        # Return in chronological order (oldest first)
        results = [self._row_to_episode(r) for r in rows]
        results.reverse()
        return results

    def _semantic_episode_search(
        self,
        query_vec: np.ndarray,
        top_k:     int,
        now:       float,
    ) -> list[EpisodeResult]:
        """
        Load all embeddings, compute composite scores, return top_k.
        This is a brute-force scan — acceptable for personal-scale memory.
        """
        rows = self._db.fetchall(
            """SELECT id, session_id, role, source, content,
                      salience_weight, created_at, access_count, embedding
                 FROM episodes
                WHERE embedding IS NOT NULL"""
        )
        if not rows:
            return []

        scored: list[tuple[float, EpisodeResult]] = []
        for row in rows:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                ep  = self._row_to_episode(row)
                ep.score = self._composite_score(query_vec, emb, ep, now)
                scored.append((ep.score, ep))
            except Exception as exc:
                logger.warning("Skipping episode %s during search: %s", row["id"], exc)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def _semantic_fact_search(
        self,
        query_vec: np.ndarray,
        top_k:     int,
        now:       float,
    ) -> list[FactResult]:
        rows = self._db.fetchall(
            """SELECT id, fact, confidence, source, created_at, embedding
                 FROM semantic_facts
                WHERE embedding IS NOT NULL"""
        )
        if not rows:
            return self._recent_facts(top_k)

        scored: list[tuple[float, FactResult]] = []
        for row in rows:
            try:
                emb  = np.frombuffer(row["embedding"], dtype=np.float32)
                sim  = float(_cosine_similarity(query_vec, emb))
                age  = now - row["created_at"]
                rec  = math.exp(-age / max(self._half_life, 1))
                score = (self._sem_w * sim) + (self._rec_w * rec)
                fr = FactResult(
                    id=row["id"], fact=row["fact"],
                    confidence=row["confidence"], source=row["source"],
                    created_at=row["created_at"], score=score
                )
                scored.append((score, fr))
            except Exception as exc:
                logger.warning("Skipping fact %s: %s", row["id"], exc)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def _recent_facts(self, limit: int) -> list[FactResult]:
        rows = self._db.fetchall(
            """SELECT id, fact, confidence, source, created_at
                 FROM semantic_facts
             ORDER BY created_at DESC
                LIMIT ?""",
            (limit,)
        )
        return [
            FactResult(
                id=r["id"], fact=r["fact"], confidence=r["confidence"],
                source=r["source"], created_at=r["created_at"]
            )
            for r in rows
        ]

    def _find_entities_in_text(self, text: str) -> list[EntityResult]:
        """Naive word-match entity lookup. Fast and good enough for personal scale."""
        if not text.strip():
            return []
        rows = self._db.fetchall(
            "SELECT id, name, type, description, importance FROM entities"
        )
        text_lower = text.lower()
        matches = [
            EntityResult(
                id=r["id"], name=r["name"], entity_type=r["type"],
                description=r["description"], importance=r["importance"]
            )
            for r in rows
            if r["name"].lower() in text_lower
        ]
        matches.sort(key=lambda e: e.importance, reverse=True)
        return matches[:5]

    # -----------------------------------------------------------------------
    # Internal — scoring and formatting
    # -----------------------------------------------------------------------

    def _composite_score(
        self,
        query_vec: np.ndarray,
        ep_vec:    np.ndarray,
        ep:        EpisodeResult,
        now:       float,
    ) -> float:
        sim     = float(_cosine_similarity(query_vec, ep_vec))
        age     = now - ep.created_at
        rec     = math.exp(-age / max(self._half_life, 1))
        sal     = ep.salience_weight * math.log1p(ep.access_count + 1) / math.log1p(10)
        return (self._sem_w * sim) + (self._rec_w * rec) + (self._sal_w * sal)

    def _trim_to_budget(
        self,
        episodes: list[EpisodeResult],
        facts:    list[FactResult],
        budget:   int,
    ) -> tuple[list[EpisodeResult], list[FactResult], int]:
        """Drop lowest-scored items until estimated token count fits budget."""
        budget_chars = budget * CHARS_PER_TOKEN
        used         = 0
        ep_out: list[EpisodeResult] = []
        ft_out: list[FactResult]    = []

        for ep in episodes:
            cost = len(ep.content) + 20   # 20 chars overhead per entry
            if used + cost <= budget_chars:
                ep_out.append(ep)
                used += cost

        for ft in facts:
            cost = len(ft.fact) + 10
            if used + cost <= budget_chars:
                ft_out.append(ft)
                used += cost

        return ep_out, ft_out, used // CHARS_PER_TOKEN

    def _format_context(
        self,
        episodes: list[EpisodeResult],
        facts:    list[FactResult],
        entities: list[EntityResult],
    ) -> str:
        """Format retrieved memory into a string for the LLM system prompt."""
        if not episodes and not facts and not entities:
            return ""

        parts: list[str] = ["[MEMORY]"]

        if facts:
            parts.append("Known facts:")
            for f in facts:
                parts.append(f"  - {f.fact}")

        if entities:
            parts.append("Known entities:")
            for e in entities:
                desc = f" ({e.description})" if e.description else ""
                parts.append(f"  - {e.name} [{e.entity_type}]{desc}")

        if episodes:
            parts.append("Relevant past exchanges:")
            for ep in episodes:
                role_label = ep.role.upper()
                parts.append(f"  [{role_label}] {ep.content}")

        parts.append("[/MEMORY]")
        return "\n".join(parts)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._embed_fn is None or not text.strip():
            return None
        try:
            vector = self._embed_fn(text)
            return np.array(vector, dtype=np.float32) if vector else None
        except Exception as exc:
            logger.warning("Reader embed failed: %s", exc)
            return None

    def _row_to_episode(self, row) -> EpisodeResult:
        return EpisodeResult(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            source=row["source"],
            content=row["content"],
            salience_weight=row["salience_weight"],
            created_at=row["created_at"],
            access_count=row["access_count"],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float32 vectors. Returns 0.0 on error."""
    try:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0
