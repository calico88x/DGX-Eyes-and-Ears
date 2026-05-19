#!/bin/bash
# eae — Eyes and Ears vLLM model control
#
# Install:
#   chmod +x eae.sh
#   sudo ln -s "$(pwd)/eae.sh" /usr/local/bin/eae
#
# Usage:
#   eae start    — load vLLM models into GPU
#   eae stop     — unload vLLM models, free GPU
#   eae status   — check running state
#
# Whisper and Kokoro are always-on system services, not managed here.

case "$1" in
  start)
    echo "Loading Eyes and Ears models..."
    sudo systemctl start vllm-chat-vision
    echo "  ✓ vllm-chat-vision — loading (~2 min)"
    sudo systemctl start vllm-embedding
    echo "  ✓ vllm-embedding — waiting for chat model to be ready"
    echo ""
    echo "  Watch progress: journalctl -u vllm-chat-vision -f"
    ;;
  stop)
    echo "Unloading Eyes and Ears models..."
    sudo systemctl stop vllm-embedding && echo "  ✓ vllm-embedding stopped"
    sudo systemctl stop vllm-chat-vision && echo "  ✓ vllm-chat-vision stopped"
    ;;
  status)
    for s in vllm-chat-vision vllm-embedding; do
      echo "── $s ──"
      sudo systemctl status "$s" --no-pager | grep -E "Active:|Loaded:"
    done
    ;;
  *)
    echo "Usage: eae {start|stop|status}"
    exit 1
    ;;
esac
