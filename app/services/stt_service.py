"""
app/services/stt_service.py

Real-time speech-to-text via WhisperLiveKit WebSocket.

Runs a background thread that:
  1. Connects to ws://host:9090/asr
  2. Reads audio chunks from an internal queue
  3. Sends them over the WebSocket
  4. Receives JSON transcripts and fires registered callbacks

Transcript callback signature:
    fn(text: str, is_final: bool, confidence: float)

Usage
-----
stt = STTService(config)
stt.on_transcript(my_callback)
stt.start()

# In your audio capture loop:
stt.send_audio(pcm_chunk)

# On shutdown:
stt.stop()
"""

import json
import logging
import queue
import threading
from typing import Callable, Optional

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool, float], None]


class STTService:
    """
    WhisperLiveKit WebSocket STT adapter.

    Audio is sent via send_audio() from the capture thread.
    Transcripts arrive via registered on_transcript() callbacks.

    The WebSocket runs in a dedicated background thread so it never
    blocks the main pipeline.
    """

    def __init__(self, config: Config):
        svc             = config.services.stt
        self._url       = svc.base_url         # ws://host:9090/asr
        self._chunk_ms  = svc.chunk_ms
        self._callbacks: list[TranscriptCallback] = []
        self._audio_q:  queue.Queue[Optional[bytes]] = queue.Queue(maxsize=50)
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt  = threading.Event()
        self._connected = False

        logger.info("STTService ready | url=%s chunk_ms=%d", self._url, self._chunk_ms)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def on_transcript(self, callback: TranscriptCallback) -> None:
        """Register a callback invoked on every transcript segment."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the WebSocket background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("STTService already running")
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target = self._run,
            name   = "stt-websocket",
            daemon = True,
        )
        self._thread.start()
        logger.info("STTService started")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        logger.info("STTService stopping...")
        self._stop_evt.set()
        self._audio_q.put(None)   # unblock the send loop

        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("STTService stopped")

    def send_audio(self, chunk: bytes) -> None:
        """
        Queue a raw PCM audio chunk for sending.
        Non-blocking — drops the chunk if the queue is full.
        Expected: 16kHz, mono, int16 PCM bytes.
        """
        try:
            self._audio_q.put_nowait(chunk)
        except queue.Full:
            logger.debug("STT audio queue full — dropping chunk")

    @property
    def connected(self) -> bool:
        return self._connected

    # -----------------------------------------------------------------------
    # Internal — WebSocket thread
    # -----------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread: connect, send audio, receive transcripts."""
        try:
            import websockets
            import asyncio
            asyncio.run(self._async_run())
        except ImportError:
            logger.error(
                "websockets not installed. Run: pip install websockets"
            )
        except Exception as exc:
            logger.error("STT WebSocket thread crashed: %s", exc)
        finally:
            self._connected = False

    async def _async_run(self) -> None:
        import asyncio
        import websockets

        logger.info("STT connecting to %s", self._url)

        try:
            async with websockets.connect(self._url) as ws:
                self._connected = True
                logger.info("STT connected")

                # Run sender and receiver concurrently
                await asyncio.gather(
                    self._send_loop(ws),
                    self._receive_loop(ws),
                )
        except Exception as exc:
            logger.error("STT WebSocket error: %s", exc)
        finally:
            self._connected = False

    async def _send_loop(self, ws) -> None:
        """Drain the audio queue and send chunks over the WebSocket."""
        import asyncio

        while not self._stop_evt.is_set():
            try:
                # Non-blocking get with short timeout so we can check stop_evt
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._audio_q.get(timeout=0.1)
                )
                if chunk is None:   # sentinel
                    break
                await ws.send(chunk)
            except Exception:
                if self._stop_evt.is_set():
                    break

    async def _receive_loop(self, ws) -> None:
        """Receive transcript JSON from the WebSocket and fire callbacks."""
        import asyncio
        import websockets

        while not self._stop_evt.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                self._handle_message(raw)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                logger.warning("STT WebSocket connection closed")
                break
            except Exception as exc:
                logger.error("STT receive error: %s", exc)
                break

    def _handle_message(self, raw: str) -> None:
        """
        Parse WhisperLiveKit transcript messages and fire callbacks.

        WhisperLiveKit format:
          {"type": "config", ...}                          ← ignore
          {"status": "no_audio_detected", "lines": [], ...} ← ignore
          {"status": "active_transcription",
           "lines": [{"speaker": -2, "text": "hello", ...}],
           "buffer_transcription": "confirmed text",
           ...}

        buffer_transcription contains finalized/confirmed text.
        lines[*].text contains the live partial transcript.
        """
        try:
            data = json.loads(raw)

            # Ignore config and non-transcription messages
            if data.get("type") == "config":
                return
            if data.get("status") not in ("active_transcription",):
                return

            # Prefer buffer_transcription (finalized) over lines (partial)
            buffer = str(data.get("buffer_transcription", "")).strip()
            if buffer:
                logger.info("Transcript | final=True text=%r", buffer[:60])
                for cb in self._callbacks:
                    try:
                        cb(buffer, True, 1.0)
                    except Exception as exc:
                        logger.error("STT callback error: %s", exc)
                return

            # Fall back to last non-empty line (partial)
            lines = data.get("lines", [])
            text = ""
            for line in lines:
                t = str(line.get("text", "")).strip()
                if t:
                    text = t

            if text:
                logger.debug("Transcript | final=False text=%r", text[:60])
                for cb in self._callbacks:
                    try:
                        cb(text, False, 1.0)
                    except Exception as exc:
                        logger.error("STT callback error: %s", exc)

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("STT malformed message: %s | raw=%r", exc, raw[:100])
