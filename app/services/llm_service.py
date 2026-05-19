"""
app/services/llm_service.py

Streaming chat completions via LiteLLM (OpenAI-compatible API).

The primary method is stream_chat() which yields text chunks as they
arrive via Server-Sent Events (SSE). The loop feeds these chunks
directly to the TTS service for low-latency speech output.

A non-streaming chat() method is also provided for cases where the
full response is needed before acting (e.g. memory summarisation).
"""

import json
import logging
from typing import Generator, Iterator, Optional

import requests

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)


class LLMService:
    """
    Streaming LLM chat adapter.

    Wraps LiteLLM's OpenAI-compatible /v1/chat/completions endpoint.
    Handles SSE parsing, error recovery, and token counting.

    Usage
    -----
    llm = LLMService(config)
    for chunk in llm.stream_chat(messages):
        tts.synthesize_chunk(chunk)
    """

    def __init__(self, config: Config):
        svc            = config.services.litellm
        self._base_url = svc.base_url.rstrip("/")
        self._api_key  = svc.api_key
        self._model    = config.models.chat
        self._params   = config.params.get("chat_llm", {})
        self._timeout  = 30

        logger.info("LLMService ready | model=%s url=%s", self._model, self._base_url)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def stream_chat(
        self,
        messages:      list[dict],
        system_prompt: Optional[str]   = None,
        image_bytes:   Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """
        Yield text chunks as they stream from the LLM.

        Parameters
        ----------
        messages      : list[dict]     Conversation history in OpenAI format.
        system_prompt : str | None     Injected as the first system message if provided.
        image_bytes   : bytes | None   JPEG frame to inject into the last user message.
                                       When provided, the VL model sees actual pixels.

        Yields
        ------
        str  Text fragment (word or partial word) as it arrives.
        """
        full_messages = self._build_messages(messages, system_prompt)
        if image_bytes:
            full_messages = self._inject_image(full_messages, image_bytes)

        payload = {
            "model":    self._model,
            "messages": full_messages,
            "stream":   True,
            **{k: v for k, v in self._params.items() if k != "stream"},
        }

        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json    = payload,
                headers = self._headers(),
                stream  = True,
                timeout = self._timeout,
            )
            response.raise_for_status()
            yield from self._parse_sse(response)

        except requests.RequestException as exc:
            logger.error("LLMService stream_chat failed: %s", exc)

    def chat(
        self,
        messages:      list[dict],
        system_prompt: Optional[str]   = None,
        image_bytes:   Optional[bytes] = None,
    ) -> str:
        """Non-streaming chat. Returns the full response as a string."""
        return "".join(self.stream_chat(messages, system_prompt, image_bytes))

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _build_messages(
        self,
        messages:      list[dict],
        system_prompt: Optional[str],
    ) -> list[dict]:
        """Prepend system message if provided and not already present."""
        if not system_prompt:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": system_prompt}] + messages

    def _inject_image(
        self,
        messages:    list[dict],
        image_bytes: bytes,
    ) -> list[dict]:
        """
        Attach a JPEG frame to the last user message as a multimodal payload.

        The VL model receives actual screen pixels alongside the text,
        giving it richer context than a text description alone.

        Only the current turn gets the image — historical messages stay
        text-only, keeping context window usage bounded.
        """
        import base64

        if not messages:
            return messages

        # Find the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        b64     = base64.b64encode(image_bytes).decode("ascii")
        text    = messages[last_user_idx].get("content", "")
        updated = messages[last_user_idx].copy()

        updated["content"] = [
            {
                "type":      "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {
                "type": "text",
                "text": text if isinstance(text, str) else "",
            },
        ]

        return messages[:last_user_idx] + [updated] + messages[last_user_idx + 1:]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    def _parse_sse(self, response: requests.Response) -> Iterator[str]:
        """Parse OpenAI-style SSE stream and yield content fragments."""
        for line in response.iter_lines():
            if not line:
                continue

            # SSE lines are bytes
            if isinstance(line, bytes):
                line = line.decode("utf-8")

            if not line.startswith("data: "):
                continue

            payload = line[6:].strip()

            if payload == "[DONE]":
                break

            try:
                data    = json.loads(payload)
                delta   = data["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
