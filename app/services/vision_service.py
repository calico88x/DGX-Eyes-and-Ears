"""
app/services/vision_service.py

Screen frame description via Qwen2.5-VL-7B-Instruct served through vLLM.

Accepts raw image bytes (JPEG/PNG), encodes them as base64, and sends
them to the vLLM OpenAI-compatible vision endpoint. Returns a short
text description of what is visible on screen.

Called by the vision pipeline on a configurable interval.
The description feeds into the classifier's salience scoring.
"""

import base64
import logging
from typing import Optional

import requests

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)

# Default prompt — tuned for screen share content
DEFAULT_PROMPT = (
    "Describe what is currently visible on the screen in 1-2 sentences. "
    "Focus on activity, notable events, and what the user appears to be doing. "
    "Be concise and factual."
)


class VisionService:
    """
    Qwen2.5-VL screen description adapter.

    Usage
    -----
    vision = VisionService(config)
    description = vision.describe_frame(jpeg_bytes)
    # → "User is playing a racing game. A car has just crashed into a barrier."
    """

    def __init__(self, config: Config):
        svc            = config.services.vision
        self._base_url = svc.base_url.rstrip("/")
        self._api_key  = config.services.litellm.api_key
        self._model    = config.models.vision
        self._params   = config.params.get("vision_model", {})
        self._timeout  = 15

        logger.info(
            "VisionService ready | model=%s url=%s", self._model, self._base_url
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def describe_frame(
        self,
        image_bytes: bytes,
        prompt:      Optional[str] = None,
        media_type:  str           = "image/jpeg",
    ) -> str:
        """
        Describe the content of a screen frame.

        Parameters
        ----------
        image_bytes : bytes       Raw image bytes (JPEG recommended for speed).
        prompt      : str | None  Custom prompt. Uses DEFAULT_PROMPT if None.
        media_type  : str         MIME type of the image.

        Returns
        -------
        str  Text description. Returns empty string on failure.
        """
        if not image_bytes:
            return ""

        b64  = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{media_type};base64,{b64}"

        payload = {
            "model":      self._model,
            "stream":     False,
            "max_tokens": self._params.get("max_tokens",  300),
            "temperature": self._params.get("temperature", 0.4),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type":      "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": prompt or DEFAULT_PROMPT,
                        },
                    ],
                }
            ],
        }

        try:
            response = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json    = payload,
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type":  "application/json",
                },
                timeout = self._timeout,
            )
            response.raise_for_status()
            description = (
                response.json()["choices"][0]["message"]["content"].strip()
            )

            logger.debug(
                "Vision described | frame_bytes=%d desc_len=%d desc=%r",
                len(image_bytes), len(description), description[:80]
            )
            return description

        except requests.RequestException as exc:
            logger.error("VisionService.describe_frame failed: %s", exc)
            return ""
        except (KeyError, IndexError) as exc:
            logger.error("VisionService response parse error: %s", exc)
            return ""
