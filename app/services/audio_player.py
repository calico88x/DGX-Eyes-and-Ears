"""
app/services/audio_player.py

Plays audio bytes through the system's default (or configured) output device.

Device selection
----------------
Call configure_audio() once at startup to set the output device.
To list available devices on Z790:

    python -c "import sounddevice; print(sounddevice.query_devices())"

Then set in config/params.yaml:
    audio:
      output_device: "Speakers (Realtek HD Audio)"   # name substring or index
      input_device:  "Microphone (USB Audio)"         # for future mic capture

Install sounddevice for real playback:
    pip install sounddevice
"""

import io
import logging
import os
import tempfile
import wave
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Module-level device config — set via configure_audio()
_output_device: Optional[Union[str, int]] = None
_input_device:  Optional[Union[str, int]] = None


def configure_audio(
    output_device: Optional[Union[str, int]] = None,
    input_device:  Optional[Union[str, int]] = None,
) -> None:
    """
    Set audio devices for playback and capture.
    Call once at startup before any play_audio() calls.

    Parameters
    ----------
    output_device : str | int | None
        Device name substring or index for playback. None = system default.
    input_device : str | int | None
        Device name substring or index for mic capture. None = system default.
    """
    global _output_device, _input_device
    _output_device = output_device
    _input_device  = input_device
    if output_device is not None:
        logger.info("Audio output device set: %r", output_device)
    if input_device is not None:
        logger.info("Audio input device set: %r", input_device)


def list_devices() -> str:
    """Return a formatted string of all available audio devices."""
    try:
        import sounddevice as sd
        return str(sd.query_devices())
    except ImportError:
        return "sounddevice not installed — pip install sounddevice"


def play_audio(audio_bytes: bytes, blocking: bool = True) -> bool:
    """
    Play WAV audio bytes through the configured output device.

    Returns True if audio played, False if skipped or unavailable.
    """
    if not audio_bytes:
        return False

    try:
        return _play_sounddevice(audio_bytes, blocking)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("sounddevice playback failed: %s", exc)

    try:
        return _play_os(audio_bytes)
    except Exception as exc:
        logger.warning("OS audio playback failed: %s", exc)

    logger.info(
        "Audio playback unavailable — %d bytes skipped. pip install sounddevice",
        len(audio_bytes)
    )
    return False


def _play_sounddevice(audio_bytes: bytes, blocking: bool) -> bool:
    import sounddevice as sd
    import numpy as np

    with wave.open(io.BytesIO(audio_bytes)) as wf:
        rate  = wf.getframerate()
        raw   = wf.readframes(wf.getnframes())
        arr   = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    sd.play(arr, samplerate=rate, device=_output_device)
    if blocking:
        sd.wait()
    return True


def _play_os(audio_bytes: bytes) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        if os.name == "nt":
            os.startfile(tmp_path)
        elif os.name == "posix":
            import subprocess
            player = "aplay" if os.path.exists("/usr/bin/aplay") else "afplay"
            subprocess.run([player, tmp_path], check=True, capture_output=True)
        return True
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
