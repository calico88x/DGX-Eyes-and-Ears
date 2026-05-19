#!/usr/bin/env python3
"""
zgx/whisper_server.py — Utterance-based faster-whisper WebSocket ASR server.

Accumulates audio while speech is happening, waits for silence, then
transcribes the complete utterance in one shot. Whisper gets full sentence
context → accurate transcription.

Configuration via environment variables (or edit the defaults below):

  WHISPER_MODEL_DIR   Path to a local CT2 model directory.
                      If unset, uses WHISPER_MODEL (HF model ID).
  WHISPER_MODEL       HuggingFace model ID (default: deepdml/faster-whisper-large-v3-turbo-ct2)
  WHISPER_DEVICE      cuda or cpu (default: cuda)
  WHISPER_COMPUTE     float16, int8_float16, int8 (default: float16)
  WHISPER_LANGUAGE    Language code (default: en)
  WHISPER_PORT        Server port (default: 9091)

Usage:
    python whisper_server.py

WebSocket: ws://0.0.0.0:{WHISPER_PORT}/asr
Accepts:   raw PCM int16 s16le at 16000 Hz
Returns:   WhisperLiveKit-compatible JSON
"""

import asyncio
import json
import logging
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("whisper_server")

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_DIR      = os.environ.get("WHISPER_MODEL_DIR", "")
MODEL_ID       = os.environ.get("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
DEVICE         = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE   = os.environ.get("WHISPER_COMPUTE", "float16")
LANGUAGE       = os.environ.get("WHISPER_LANGUAGE", "en")
PORT           = int(os.environ.get("WHISPER_PORT", "9091"))
SAMPLE_RATE    = 16000

# Utterance detection
# ENERGY_THRESHOLD : RMS amplitude above which a chunk counts as speech.
#   Raise if background noise triggers false starts.
#   Lower if your mic is quiet and speech isn't being detected.
ENERGY_THRESHOLD       = int(os.environ.get("WHISPER_ENERGY_THRESHOLD", "400"))
SILENCE_DURATION       = float(os.environ.get("WHISPER_SILENCE_DURATION", "0.7"))
MIN_SPEECH_DURATION    = 0.4
MAX_UTTERANCE_DURATION = 30.0

# ── Load model ────────────────────────────────────────────────────────────────

model_ref = MODEL_DIR if MODEL_DIR else MODEL_ID
logger.info("Loading faster-whisper model: %s", model_ref)
logger.info("Device: %s  Compute: %s", DEVICE, COMPUTE_TYPE)

from faster_whisper import WhisperModel  # noqa: E402

try:
    model = WhisperModel(model_ref, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded successfully")
except Exception as exc:
    logger.warning("Primary load failed (%s) — retrying with int8", exc)
    model = WhisperModel(model_ref, device=DEVICE, compute_type="int8")
    logger.info("Model loaded with int8 fallback")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Whisper ASR Server")


@app.get("/health")
def health():
    return {"status": "ok", "backend": "faster-whisper", "ready": True}


@app.websocket("/asr")
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected: %s", ws.client)
    await ws.send_text(json.dumps({"type": "config", "useAudioWorklet": True, "mode": "full"}))

    audio_buffer   = np.array([], dtype=np.int16)
    silence_frames = 0
    is_speaking    = False
    silence_limit  = int(SILENCE_DURATION * SAMPLE_RATE)
    min_samples    = int(MIN_SPEECH_DURATION * SAMPLE_RATE)
    max_samples    = int(MAX_UTTERANCE_DURATION * SAMPLE_RATE)
    loop           = asyncio.get_event_loop()

    try:
        while True:
            raw   = await ws.receive_bytes()
            chunk = np.frombuffer(raw, dtype=np.int16)

            energy    = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            is_speech = energy > ENERGY_THRESHOLD

            if is_speech:
                audio_buffer   = np.append(audio_buffer, chunk)
                silence_frames = 0
                is_speaking    = True

            elif is_speaking:
                audio_buffer   = np.append(audio_buffer, chunk)
                silence_frames += len(chunk)

                end_of_utterance = (
                    silence_frames >= silence_limit
                    or len(audio_buffer) >= max_samples
                )

                if end_of_utterance:
                    utterance      = audio_buffer.copy()
                    audio_buffer   = np.array([], dtype=np.int16)
                    silence_frames = 0
                    is_speaking    = False

                    if len(utterance) < min_samples:
                        continue

                    audio_f32 = utterance.astype(np.float32) / 32768.0

                    def _transcribe(a=audio_f32):
                        segs, _ = model.transcribe(a, language=LANGUAGE, vad_filter=False)
                        return " ".join(s.text.strip() for s in segs).strip()

                    text = await loop.run_in_executor(None, _transcribe)

                    if text:
                        logger.info("Transcript: %r", text)
                        msg = {
                            "status":               "active_transcription",
                            "lines":                [{"speaker": -2, "text": text,
                                                      "start": "0:00:00.00",
                                                      "end":   "0:00:01.00"}],
                            "buffer_transcription": text,
                            "buffer_diarization":   "",
                            "buffer_translation":   "",
                            "remaining_time_transcription": 0,
                            "remaining_time_diarization":   0,
                        }
                        await ws.send_text(json.dumps(msg))

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", ws.client)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
