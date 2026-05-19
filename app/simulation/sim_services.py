"""
app/simulation/sim_services.py

Offline simulation stubs for all four pipeline services.
Identical interfaces to the real services — no network calls.

    SimLLMService     → replaces LLMService
    SimTTSService     → replaces TTSService
    SimSTTService     → replaces STTService
    SimVisionService  → replaces VisionService

EmbeddingService has no sim stub — the memory system handles
the no-embed-fn case natively (recency-only retrieval).

Usage in the main loop
----------------------
if config.sim_mode:
    from app.simulation.sim_services import (
        SimLLMService as LLMService,
        SimTTSService as TTSService,
        SimSTTService as STTService,
        SimVisionService as VisionService,
    )
else:
    from app.services.llm_service    import LLMService
    from app.services.tts_service    import TTSService
    from app.services.stt_service    import STTService
    from app.services.vision_service import VisionService
"""

import logging
import struct
import threading
import time
from typing import Callable, Generator, Iterator, Optional

from app.utils.config_loader import Config

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool, float], None]


# ---------------------------------------------------------------------------
# SimLLMService
# ---------------------------------------------------------------------------

class SimLLMService:
    """
    Simulated LLM. Streams canned responses word-by-word with configurable delay.

    Matches canned responses by checking if a trigger phrase appears in
    the last user message. Falls back to the default response if no match.

    Parameters
    ----------
    config            : Config
    word_delay_s      : float   Pause between words (simulates token stream).
    first_token_delay_s : float Pause before first word (simulates THINKING time).
    """

    DEFAULT_RESPONSES = [
        "That's interesting! Tell me more.",
        "I see what you mean.",
        "Got it. What else is on your mind?",
        "Hmm, that's a good point.",
        "I'm following along — keep going.",
    ]

    def __init__(
        self,
        config:               Config,
        word_delay_s:         float = 0.05,
        first_token_delay_s:  float = 0.3,
    ):
        self._params           = config.params.get("chat_llm", {})
        self._word_delay       = word_delay_s
        self._first_token_delay = first_token_delay_s
        self._canned           = list(config.simulation.canned_responses)
        self._response_cycle   = 0

        logger.info(
            "SimLLMService ready | word_delay=%.2fs canned=%d",
            word_delay_s, len(self._canned)
        )

    def stream_chat(
        self,
        messages:      list[dict],
        system_prompt: Optional[str]   = None,
        image_bytes:   Optional[bytes] = None,
    ) -> Generator[str, None, None]:
        """Yield words from a canned response with simulated latency."""
        response = self._pick_response(messages)

        time.sleep(self._first_token_delay)

        for word in response.split(" "):
            yield word + " "
            time.sleep(self._word_delay)

    def chat(
        self,
        messages:      list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        return "".join(self.stream_chat(messages, system_prompt))

    def _pick_response(self, messages: list[dict]) -> str:
        # Check canned triggers against last user message
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            ""
        )
        last_user_lower = last_user.lower()

        for canned in self._canned:
            if canned.trigger.lower() in last_user_lower:
                return canned.response

        # Cycle through default responses
        response = self.DEFAULT_RESPONSES[self._response_cycle % len(self.DEFAULT_RESPONSES)]
        self._response_cycle += 1
        return response


# ---------------------------------------------------------------------------
# SimTTSService
# ---------------------------------------------------------------------------

class SimTTSService:
    """
    Simulated TTS. Returns valid silent WAV bytes so downstream audio
    handling doesn't need special-casing for sim mode.

    Duration scales with text length (100ms per word, min 200ms).
    """

    def __init__(self, config: Config):
        self._delay_s = config.simulation.vision_delay_ms / 1000.0 * 0.3
        logger.info("SimTTSService ready")

    def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        if not text.strip():
            return _make_silence_wav(0.1)

        words      = len(text.split())
        duration_s = max(0.2, words * 0.1)

        time.sleep(min(self._delay_s, 0.1))   # small sim latency

        audio = _make_silence_wav(duration_s)
        logger.debug(
            "SimTTS synthesized | words=%d duration=%.1fs bytes=%d",
            words, duration_s, len(audio)
        )
        return audio


# ---------------------------------------------------------------------------
# SimSTTService
# ---------------------------------------------------------------------------

class SimSTTService:
    """
    Simulated STT. Fires canned transcript callbacks on a timer.

    Useful for testing the full pipeline loop offline — the sim STT
    behaves like a real user speaking at configured intervals.

    Parameters
    ----------
    config           : Config
    phrases          : list[str] | None   Phrases to cycle through.
    phrase_interval_s : float             How often to emit a phrase.
    """

    DEFAULT_PHRASES = [
        "hey what are you seeing on my screen",
        "what do you think about this",
        "can you help me understand what's happening here",
        "this looks interesting right",
        "what should I do next",
    ]

    def __init__(
        self,
        config:            Config,
        phrases:           Optional[list[str]] = None,
        phrase_interval_s: float = 4.0,
    ):
        self._delay_ms      = config.simulation.stt_delay_ms
        self._phrases       = phrases or list(self.DEFAULT_PHRASES)
        self._interval_s    = phrase_interval_s
        self._callbacks:    list[TranscriptCallback] = []
        self._thread:       Optional[threading.Thread] = None
        self._stop_evt      = threading.Event()
        self._phrase_idx    = 0
        self._connected     = False

        logger.info(
            "SimSTTService ready | phrases=%d interval=%.1fs",
            len(self._phrases), phrase_interval_s
        )

    def on_transcript(self, callback: TranscriptCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target = self._run,
            name   = "sim-stt",
            daemon = True,
        )
        self._thread.start()
        self._connected = True
        logger.info("SimSTTService started")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._connected = False
        logger.info("SimSTTService stopped")

    def send_audio(self, chunk: bytes) -> None:
        pass   # no-op in sim — audio is internally generated

    def inject_phrase(self, text: str, is_final: bool = True, confidence: float = 0.95) -> None:
        """Manually fire a transcript callback. Useful in tests."""
        for cb in self._callbacks:
            try:
                cb(text, is_final, confidence)
            except Exception as exc:
                logger.error("SimSTT callback error: %s", exc)

    @property
    def connected(self) -> bool:
        return self._connected

    def _run(self) -> None:
        time.sleep(1.0)   # small startup delay
        while not self._stop_evt.is_set():
            phrase = self._phrases[self._phrase_idx % len(self._phrases)]
            self._phrase_idx += 1

            # Simulate partial transcript first
            time.sleep(self._delay_ms / 1000.0)
            self.inject_phrase(phrase.split()[0] + "...", is_final=False, confidence=0.7)

            time.sleep(0.5)
            self.inject_phrase(phrase, is_final=True, confidence=0.95)

            # Wait before next phrase
            self._stop_evt.wait(timeout=self._interval_s)


# ---------------------------------------------------------------------------
# SimVisionService
# ---------------------------------------------------------------------------

class SimVisionService:
    """
    Simulated vision service. Returns canned screen descriptions
    cycling through a scenario list.

    Descriptions are designed to produce a range of classifier outputs
    so the full pipeline exercises all state transitions in sim mode.
    """

    DEFAULT_DESCRIPTIONS = [
        "User is browsing a code editor with Python files open.",
        "A terminal window is visible with a running process.",
        "User appears to be reading documentation in a web browser.",
        "A racing game is on screen. The car is navigating a corner.",
        "User is watching a video. A presenter is speaking on screen.",
        "Desktop is visible with multiple windows open. Nothing notable.",
        "A chat application is open. User appears to be reading messages.",
        "User is browsing a website. Text and images are visible.",
    ]

    def __init__(
        self,
        config:       Config,
        descriptions: Optional[list[str]] = None,
    ):
        self._descriptions = descriptions or list(self.DEFAULT_DESCRIPTIONS)
        self._delay_s      = config.simulation.vision_delay_ms / 1000.0
        self._idx          = 0

        logger.info(
            "SimVisionService ready | descriptions=%d", len(self._descriptions)
        )

    def describe_frame(
        self,
        image_bytes: bytes,
        prompt:      Optional[str] = None,
        media_type:  str           = "image/jpeg",
    ) -> str:
        time.sleep(self._delay_s * 0.1)   # minimal latency

        description = self._descriptions[self._idx % len(self._descriptions)]
        self._idx  += 1

        logger.debug("SimVision described | desc=%r", description[:60])
        return description

    def set_description(self, description: str) -> None:
        """Fix the next response to a specific description. Useful in tests."""
        self._descriptions.insert(self._idx, description)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _make_silence_wav(duration_s: float, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file containing silence of the given duration."""
    num_samples  = int(sample_rate * duration_s)
    data_size    = num_samples * 2   # 16-bit mono
    header_size  = 44
    file_size    = header_size + data_size - 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        file_size,
        b"WAVE",
        b"fmt ",
        16,            # chunk size
        1,             # PCM
        1,             # mono
        sample_rate,
        sample_rate * 2,   # byte rate
        2,             # block align
        16,            # bits per sample
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size
