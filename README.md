# Eyes and Ears

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Server-NVIDIA_DGX_Spark-76B900?style=flat&logo=nvidia&logoColor=white)
![Server](https://img.shields.io/badge/platform-aarch64%20Ubuntu-orange)
![Client](https://img.shields.io/badge/Client-Windows-0078D4?style=flat&logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Tests](https://img.shields.io/badge/Tests-370_passing-brightgreen?style=flat)

**An AI companion that sees your screen and hears your voice.**  
Watches what you're doing, reacts to what's on screen, responds to speech, and remembers across conversations.

</div>

---

## How it works

```
You speak  →  Whisper (GPU)     →  transcript
Screen     →  JPEG frame        →  VL model sees pixels directly
Both       →  single inference  →  companion responds with full context
```

When you speak, the current screen frame is injected directly into the LLM request alongside your words — the model sees **actual pixels**, not a text description. One inference, full context.

When you're quiet, the companion watches passively and reacts to interesting screen changes every ~8 seconds.

---

## Architecture

```
PC (Windows client)                    DGX Spark (inference server)
───────────────────                    ────────────────────────────
main.py                                vllm-chat-vision  :8012
  ├── MicService    ─────────────────► whisper-asr       :9091
  ├── VisionLoop    (mss screenshot)   vllm-embedding    :8013
  ├── LLMService    ─────────────────► LiteLLM           :4000
  ├── TTSService    ─────────────────► Kokoro TTS        :8880
  ├── MemoryDB      (SQLite)
  └── Dashboard     (FastAPI :7860)
```

### Key design decisions

| Decision | Reason |
|---|---|
| Raw JPEG injected into LLM on speech | Model sees actual pixels, not a text summary — more accurate |
| Rule-based classifier on CPU | Avoids GPU competition with inference models |
| Utterance-based STT | Waits for silence before transcribing — sends complete sentences |
| Vision as ambient context | Frame persists between turns so the LLM always has screen awareness |
| Single VL model for chat + vision | Qwen2.5-VL handles both — one model, less GPU memory |

---

## Requirements

### PC (Windows client)
- Python 3.11+
- Microphone and speakers
- See `requirements.txt`

### DGX Spark (inference server)
- NVIDIA DGX Spark or compatible ARM64 + Blackwell system
- CUDA 13.0
- ~51 GB GPU memory when Eyes and Ears models are loaded
- See [`server/README.md`](server/README.md) for complete setup

---

## Quick start

### 1. Clone and install (PC)

```bash
git clone https://github.com/calico88x/DGX-Eyes-and-Ears.git
cd DGX-Eyes-and-Ears
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
cp config/params.example.yaml config/params.yaml
```

**`config/config.yaml`** — set your DGX Spark IP:
```yaml
services:
  litellm:
    base_url: "http://YOUR_DGX_IP:4000"
```

**`config/params.yaml`** — find and set your audio device IDs:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```
```yaml
audio:
  output_device: 16   # your speakers/headphones
  input_device: 1     # your microphone
```

**`config/personality.yaml`** — give your companion a name and voice:
```yaml
name: "Aria"
system_prompt: |
  You are Aria, a conversational AI companion.
  React naturally like a friend watching alongside — brief, genuine, curious.
```

### 3. Set up the DGX Spark server

> See [`server/README.md`](server/README.md) for the full step-by-step guide.

**Replace the username in server files first:**
```bash
find server/ -type f | xargs sed -i "s|/home/nova|/home/$(whoami)|g"
```

**Download models:**
```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct               # ~16 GB  chat + vision
huggingface-cli download BAAI/bge-m3                               # ~570 MB  embedding
huggingface-cli download deepdml/faster-whisper-large-v3-turbo-ct2  # ~1.5 GB  STT
```

**Kokoro TTS** is also required → https://github.com/remsky/Kokoro-FastAPI

### 4. Run

On DGX Spark:
```bash
eae start        # loads vLLM models (~2 min to warm up)
```

On PC:
```bash
python main.py
```

Open **http://localhost:7860** for the real-time dashboard.

```bash
# Stop
Ctrl+C           # PC
eae stop         # DGX Spark
```

---

## Configuration files

| File | Committed | Purpose |
|---|---|---|
| `config/personality.yaml` | ✅ | Companion name, persona, system prompt |
| `config/simulation.yaml` | ✅ | Sim mode toggle |
| `config/config.example.yaml` | ✅ | Template — copy and fill in |
| `config/params.example.yaml` | ✅ | Template — copy and fill in |

---

## Simulation mode

Run the full pipeline locally without any DGX Spark services — useful for development and testing.

```yaml
# config/simulation.yaml
enabled: true
```

```bash
python main.py
```

All services are stubbed with canned responses. **370 tests** cover the complete pipeline in sim mode.

---

## Models

| Role | Model | Port |
|---|---|---|
| Chat + Vision | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 8012 |
| Embedding | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 8013 |
| Speech-to-text | [faster-whisper-large-v3-turbo](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | 9091 |
| Text-to-speech | [Kokoro](https://github.com/remsky/Kokoro-FastAPI) | 8880 |

---

## Memory

The companion maintains a persistent SQLite memory database across sessions. Each conversation is stored with embeddings and semantic facts — allowing the companion to reference shared history naturally over time.

Memory lives in `data/memory.db` and is **gitignored** — it stays on your machine only.

---

## License
