"""
app/services/embedding_service.py

Text embeddings via LiteLLM's /v1/embeddings endpoint.

Returns a float list compatible with the memory system's EmbedFn type.
Plugs directly into MemoryWriter and MemoryReader as the embed_fn argument.

Usage
-----
emb  = EmbeddingService(config)
vec  = emb.embed("user enjoys playing racing games")
# → [0.021, -0.143, ...]  (1536 floats)

# Plug into memory:
writer = MemoryWriter(db, embed_fn=emb.embed)
reader = MemoryReader(db, embed_fn=emb.embed)
"""

import logging
from typing import Optional

import requests

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    LiteLLM embedding adapter.

    The embed() method matches the EmbedFn signature expected by
    MemoryWriter and MemoryReader so it can be passed in directly.
    """

    def __init__(self, config: Config):
        svc            = config.services.litellm
        self._base_url = svc.base_url.rstrip("/")
        self._api_key  = svc.api_key
        self._model    = config.models.embedding
        self._timeout  = 10

        logger.info(
            "EmbeddingService ready | model=%s url=%s", self._model, self._base_url
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def embed(self, text: str) -> Optional[list[float]]:
        """
        Embed a text string.

        Parameters
        ----------
        text : str  Text to embed.

        Returns
        -------
        list[float] | None
            Embedding vector. Returns None on failure so the memory
            system can degrade gracefully to recency-only retrieval.
        """
        if not text.strip():
            return None

        payload = {
            "model": self._model,
            "input": text.strip(),
        }

        try:
            response = requests.post(
                f"{self._base_url}/v1/embeddings",
                json    = payload,
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type":  "application/json",
                },
                timeout = self._timeout,
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]

            logger.debug(
                "Embedding generated | text_len=%d dim=%d", len(text), len(vector)
            )
            return vector

        except requests.RequestException as exc:
            logger.error("EmbeddingService.embed request failed: %s", exc)
            return None
        except (KeyError, IndexError) as exc:
            logger.error("EmbeddingService.embed parse error: %s", exc)
            return None
