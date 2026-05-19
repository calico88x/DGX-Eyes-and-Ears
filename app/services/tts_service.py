"""
app/services/tts_service.py

Text-to-speech via Kokoro's OpenAI-compatible /v1/audio/speech endpoint.

Returns raw audio bytes (mp3). The loop is responsible for playing them.

Kokoro API:
  POST /v1/audio/speech
  {"model": "kokoro", "input": "...", "voice": "af_heart", "response_format": "mp3"}
  → audio/mpeg bytes
"""

import logging
from typing import Optional

import requests

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)

# 0.5s of silence as WAV — returned on error so the pipeline doesn't break
_SILENCE_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


class TTSService:
    """
    Kokoro TTS adapter.

    Usage
    -----
    tts = TTSService(config)
    audio_bytes = tts.synthesize("Hello, how are you?")
    # play audio_bytes through your audio output
    """

    def __init__(self, config: Config):
        svc            = config.services.tts
        self._base_url = svc.base_url.rstrip("/")
        self._voice    = svc.voice
        self._timeout  = 20

        logger.info("TTSService ready | url=%s voice=%s", self._base_url, self._voice)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Convert text to audio bytes.

        Parameters
        ----------
        text  : str          Text to synthesize.
        voice : str | None   Override the configured voice.

        Returns
        -------
        bytes  Audio bytes (mp3). Returns silence on failure.
        """
        if not text.strip():
            return _SILENCE_WAV

        payload = {
            "model":           "kokoro",
            "input":           text.strip(),
            "voice":           voice or self._voice,
            "response_format": "wav",   # sounddevice needs WAV, not MP3
        }

        try:
            response = requests.post(
                f"{self._base_url}/v1/audio/speech",
                json    = payload,
                headers = {"Content-Type": "application/json"},
                timeout = self._timeout,
            )
            response.raise_for_status()
            audio = response.content

            logger.debug(
                "TTS synthesized | chars=%d audio_bytes=%d", len(text), len(audio)
            )
            return audio

        except requests.RequestException as exc:
            logger.error("TTSService.synthesize failed: %s", exc)
            return _SILENCE_WAV
