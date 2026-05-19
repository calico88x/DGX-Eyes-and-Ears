# Eyes and Ears — Server Setup (DGX Spark / Blackwell)

Complete setup guide for the DGX Spark inference server.

## Prerequisites

- NVIDIA DGX Spark (GB10 / Blackwell) running Ubuntu 24.04
- CUDA 13.0 at `/usr/local/cuda-13.0`
- Conda/Miniforge installed
- `huggingface-cli` available (`pip install huggingface-hub`)

---

## Step 0 — Replace username

Every path in the service files uses `nova` as the username.
Replace it with your own before copying any files:

```bash
YOUR_USER=$(whoami)
# Run from repo root
find server/ -type f | xargs sed -i "s|/home/nova|/home/$YOUR_USER|g"
```

---

## Step 1 — Download models

```bash
pip install huggingface-hub

# Chat + Vision (~16 GB)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

# Embedding (~570 MB)
huggingface-cli download BAAI/bge-m3

# Whisper speech-to-text (~1.5 GB, CTranslate2 format)
huggingface-cli download deepdml/faster-whisper-large-v3-turbo-ct2
```

---

## Step 2 — Whisper server dependencies

```bash
pip install faster-whisper fastapi "uvicorn[standard]" numpy python-multipart
```

### ctranslate2 with CUDA (required for GPU transcription)

The PyPI ARM64 wheel is CPU-only. Build from source:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
pip install cmake ninja

git clone https://github.com/OpenNMT/CTranslate2.git --recursive
cd CTranslate2
mkdir build && cd build
cmake .. \
    -DWITH_CUDA=ON -DCUDA_DYNAMIC_LOADING=ON \
    -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
cd ../python && pip install .

# Verify
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

---

## Step 3 — vLLM

```bash
mkdir -p ~/venvs
python3 -m venv ~/venvs/vllm
~/venvs/vllm/bin/pip install vllm==0.20.1
~/venvs/vllm/bin/vllm --version   # should print 0.20.1
```

---

## Step 4 — LiteLLM

```bash
python3 -m venv ~/litellm-env
~/litellm-env/bin/pip install "litellm[proxy]"

mkdir -p ~/litellm
cp server/litellm_config.example.yaml ~/litellm/litellm_config.yaml
```

Run as a systemd service (create `/etc/systemd/system/litellm.service`):

```ini
[Unit]
Description=LiteLLM Proxy
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/litellm-env/bin/litellm --config /home/YOUR_USERNAME/litellm/litellm_config.yaml --port 4000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable litellm
sudo systemctl start litellm
```

---

## Step 5 — Kokoro TTS

Kokoro is an OpenAI-compatible TTS server. Install and run on port 8880.

Official repo and install instructions:
**https://github.com/remsky/Kokoro-FastAPI**

Kokoro must expose `POST /v1/audio/speech` on port `8880`.
Run it as a systemd service so it starts on boot.

---

## Step 6 — Deploy server files

```bash
cp server/whisper_server.py ~/whisper_server.py

chmod +x server/eae.sh
sudo ln -s "$(realpath server/eae.sh)" /usr/local/bin/eae

sudo cp server/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Whisper — always on
sudo systemctl enable whisper-asr
sudo systemctl start whisper-asr

# vLLM — manual via eae (loads only when needed)
```

---

## Step 7 — Firewall

```bash
sudo ufw allow 4000/tcp   # LiteLLM
sudo ufw allow 8880/tcp   # Kokoro TTS
sudo ufw allow 9091/tcp   # Whisper ASR
# vLLM ports 8012/8013 are localhost-only — do not expose
```

---

## Daily usage

```bash
eae start     # load vLLM models (~2 min)
eae stop      # unload, free GPU
eae status    # check running state

journalctl -u whisper-asr -f
journalctl -u vllm-chat-vision -f
journalctl -u litellm -f
```

---

## Port reference

| Port | Service | Auto-start |
|------|---------|------------|
| 4000 | LiteLLM proxy | Yes |
| 8012 | vLLM — Qwen2.5-VL-7B | `eae start` |
| 8013 | vLLM — bge-m3 | `eae start` |
| 8880 | Kokoro TTS | Yes |
| 9091 | Whisper ASR | Yes |

---

## GPU memory budget (128 GB unified)

| Model | VRAM |
|-------|------|
| Qwen2.5-VL-7B (0.35) | ~42 GB |
| bge-m3 (0.05) | ~6 GB |
| Whisper large-v3-turbo | ~3 GB |
| **Total** | **~51 GB** |

---

## Troubleshooting

**Whisper returns empty transcripts** — mic energy below threshold.
Adjust `WHISPER_ENERGY_THRESHOLD` env var in the whisper-asr service (default: 400).

**LLM ignores images / says screen is white** — `supports_vision: true` missing
from `litellm_config.yaml`. Add it to both VL model entries and restart LiteLLM.

**vLLM fails with memory error** — another process holds GPU memory.
Check `nvidia-smi` and free it before `eae start`.

**ctranslate2 CUDA not found after build** — run `sudo ldconfig` to refresh
the shared library cache.