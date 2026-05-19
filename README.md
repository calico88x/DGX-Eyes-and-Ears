# Eyes and Ears

An AI companion that can see your screen and hear your voice. It watches what
you're doing, reacts to what's on screen, responds to speech, and maintains
memory across conversations.

## Specifically designed for DGX Spark for inference server

```
You speak  →  Whisper (GPU)  →  transcript
Screen     →  JPEG frame     →  VL model sees pixels directly
Both       →  one inference  →  companion responds with full context
```

## Architecture

```
PC (Windows client)           DGX Spark (Linux inference server)
─────────────────────           ────────────────────────────
main.py                         vllm-chat-vision  :8012
  ├── MicService ──────────────►  whisper-asr      :9091
  ├── VisionLoop (mss)          vllm-embedding    :8013
  ├── LLMService ──────────────►  LiteLLM          :4000
  ├── TTSService ──────────────►  Kokoro TTS       :8880
  ├── MemoryDB (SQLite)
  └── UI (FastAPI :7860)
```

**Key design decisions:**
- Vision captures a frame every 8s and keeps it as ambient context (Configurable)
- When you speak, the raw JPEG is injected directly into the LLM request — the VL model sees actual pixels, not a text description
- Vision-triggered reactions use the text description path (classifier needs text to score)
- Rule-based classifier runs on CPU — no GPU competition with inference
- Whisper transcription is utterance-based: waits for silence, then sends the full sentence

## Requirements

### PC (Windows client)
- Python 3.11+
- See `requirements.txt`

### DGX Spark (Linux inference server)
- NVIDIA DGX Spark or compatible ARM64 + Blackwell
- CUDA 13.0
- vLLM, LiteLLM, Kokoro TTS
- See `server/README.md` for full setup

## Setup

### 1. Clone and install (PC)

```bash
git clone https://github.com/calico88x/dgx-eyes-and-ears.git
cd eyes-and-ears
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
cp config/params.example.yaml config/params.yaml
```

Edit `config/config.yaml` — set `YOUR_DGX Spark_IP` to your inference server's IP address.

Edit `config/params.yaml` — set audio device IDs for your microphone and speakers:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Edit `config/personality.yaml` — give your companion a name and personality.

### 3. Set up DGX Spark server

See [`server/README.md`](server/README.md) for full DGX Spark setup including vLLM, ctranslate2
CUDA build, and systemd services.

**Before anything else — replace the username in server files:**
```bash
find server/ -type f | xargs sed -i "s|/home/nova|/home/$(whoami)|g"
```

**Download models on the DGX Spark:**
```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct   # ~16 GB
huggingface-cli download BAAI/bge-m3                    # ~570 MB
huggingface-cli download deepdml/faster-whisper-large-v3-turbo-ct2  # ~1.5 GB
```

Kokoro TTS is also required on the DGX Spark.
See https://github.com/remsky/Kokoro-FastAPI for installation.

### 4. Run

On DGX Spark:
```bash
eae start        # loads vLLM models (~2 min)
```

On PC:
```bash
python main.py
```

Open `http://localhost:7860` for the real-time dashboard.

To stop:
```bash
# PC
Ctrl+C

# DGX Spark
eae stop
```

## Configuration

| File | Purpose |
|---|---|
| `config/config.yaml` | Service URLs, model names, vision intervals |
| `config/params.yaml` | LLM params, audio device IDs, classifier thresholds |
| `config/personality.yaml` | Companion name, persona, system prompt |
| `config/simulation.yaml` | Sim mode toggle (set `enabled: false` for live) |

## Simulation mode

Run without any DGX Spark services for development:

```yaml
# config/simulation.yaml
enabled: true
```

```bash
python main.py --sim
```

All services are stubbed — LLM, TTS, STT, and vision return canned responses.
370 tests cover the full pipeline in sim mode.

## Models used

| Role | Model | Served by |
|---|---|---|
| Chat + Vision | Qwen/Qwen2.5-VL-7B-Instruct | vLLM :8012 |
| Embedding | BAAI/bge-m3 | vLLM :8013 |
| Speech-to-text | faster-whisper-large-v3-turbo | whisper_server.py :9091 |
| Text-to-speech | Kokoro | Kokoro :8880 |

## License

MIT
