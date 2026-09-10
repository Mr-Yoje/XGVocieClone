#!/usr/bin/env bash
# Google Colab / 云端 GPU 环境：不装 conda，沿用系统里已有的 CUDA PyTorch。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COSY_DIR="${ROOT}/third_party/CosyVoice"
MODEL_DIR="${ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-huggingface}"

log() { printf '\n==> %s\n' "$*"; }

log "系统依赖 (sox / ffmpeg)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq sox libsox-dev ffmpeg git-lfs >/dev/null

if [[ ! -d "${COSY_DIR}/.git" ]]; then
  log "克隆 FunAudioLLM/CosyVoice"
  mkdir -p "${ROOT}/third_party"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSY_DIR"
else
  log "更新 CosyVoice 子模块"
  git -C "$COSY_DIR" submodule update --init --recursive
fi

log "安装 Python 依赖（跳过 torch/torchaudio，保留 Colab 预装 CUDA 版）"
REQ_FILTERED="$(mktemp)"
grep -viE '^(torch|torchaudio|torchvision)([=<>! ]|$)' "${COSY_DIR}/requirements.txt" > "$REQ_FILTERED" || true
python -m pip install -U pip
python -m pip install -r "$REQ_FILTERED"
python -m pip install modelscope huggingface_hub gradio
rm -f "$REQ_FILTERED"

if [[ ! -f "${MODEL_DIR}/llm.rl.pt" ]]; then
  log "下载 Fun-CosyVoice3-0.5B-2512（含 llm.rl.pt），约 7GB+"
  python "${ROOT}/download_models.py" --out "$MODEL_DIR" --source "$DOWNLOAD_SOURCE"
else
  log "已有模型权重，跳过下载: ${MODEL_DIR}"
fi

log "Colab 环境就绪。"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY
