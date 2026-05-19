"""
app/services/mic_service.py

Captures microphone audio and streams it to the STT service.

Runs in a background thread using sounddevice. Audio is captured
in chunks (default 250ms) and forwarded to stt.send_audio() which
queues it for transmission to WhisperLiveKit over WebSocket.

Usage
-----
mic = MicService(input_device=1, on_audio=stt.send_audio)
mic.start()
# ... pipeline runs ...
mic.stop()
"""

import logging
import threading
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class MicService:
    """
    Real-time microphone capture.

    Captures audio from the configured input device and calls on_audio()
    with each chunk as raw int16 PCM bytes (16kHz, mono).

    Parameters
    ----------
    input_device  : int | str | None   sounddevice device index or name.
    sample_rate   : int                Default 16000Hz (required by Whisper).
    chunk_ms      : int                Chunk size in ms. Default 250ms.
    on_audio      : Callable           Called with bytes chunk on each capture.
    """

    def __init__(
        self,
        input_device:  Optional[Union[int, str]] = None,
        sample_rate:   int = 16000,
        chunk_ms:      int = 250,
        on_audio:      Optional[Callable[[bytes], None]] = None,
    ):
        self._device      = input_device
        self._sample_rate = sample_rate
        self._chunk_ms    = chunk_ms
        self._chunk_size  = int(sample_rate * chunk_ms / 1000)
        self._on_audio    = on_audio
        self._thread:     Optional[threading.Thread] = None
        self._stop_evt    = threading.Event()
        self._active      = False

        logger.info(
            "MicService ready | device=%s sample_rate=%d chunk_ms=%d",
            input_device, sample_rate, chunk_ms
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start microphone capture in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("MicService already running")
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target = self._run,
            name   = "mic-capture",
            daemon = True,
        )
        self._thread.start()
        logger.info("MicService started")

    def stop(self) -> None:
        """Signal capture to stop and wait for the thread."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._active = False
        logger.info("MicService stopped")

    @property
    def active(self) -> bool:
        return self._active

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("sounddevice not installed — pip install sounddevice")
            return

        def _callback(indata, frames, time, status):
            if status:
                logger.debug("Mic status: %s", status)
            if self._on_audio and not self._stop_evt.is_set():
                # Convert float32 → int16 PCM bytes
                pcm = (indata[:, 0] * 32767).astype(np.int16)
                self._on_audio(pcm.tobytes())

        try:
            self._active = True
            logger.info("Mic capturing | device=%s", self._device)

            with sd.InputStream(
                device     = self._device,
                samplerate = self._sample_rate,
                channels   = 1,
                dtype      = "float32",
                blocksize  = self._chunk_size,
                callback   = _callback,
            ):
                self._stop_evt.wait()   # blocks until stop() is called

        except Exception as exc:
            logger.error("MicService capture failed: %s", exc)
        finally:
            self._active = False
