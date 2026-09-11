#!/usr/bin/env bash
# Google Colab / 云端 GPU：沿用预装 CUDA PyTorch，只装推理依赖。
# 官方 requirements 钉死 grpcio==1.57.0 / deepspeed，在 Python 3.12+ 无法构建。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COSY_DIR="${ROOT}/third_party/CosyVoice"
MODEL_DIR="${ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-huggingface}"
REQ_FILTERED="$(mktemp)"

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

log "生成推理依赖列表（跳过 torch / grpcio / deepspeed 等，并在 Py3.12+ 放开版本钉）"
python - "$COSY_DIR/requirements.txt" "$REQ_FILTERED" <<'PY'
import re
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
skip = re.compile(
    r"^(torch|torchaudio|torchvision|grpcio|deepspeed|tensorrt|vllm|nvidia-)",
    re.I,
)
py312 = sys.version_info >= (3, 12)
out = []
for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("--"):
        continue
    name = re.split(r"[<>=!; \[]", line, 1)[0]
    if skip.match(name):
        print(f"skip {name}", flush=True)
        continue
    if py312:
        marker = ""
        pkg = line
        if ";" in line:
            pkg, marker = line.split(";", 1)
            marker = ";" + marker.strip()
        pkg = re.split(r"[=<>!~]", pkg, 1)[0].strip()
        line = pkg + marker
    out.append(line)
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"kept {len(out)} packages -> {dst}", flush=True)
PY

log "安装 Python 依赖"
# Colab 预装 torch 要求 setuptools<82，不要把 setuptools 升到 84
python -m pip install -U pip wheel
python -m pip install "setuptools>=70,<82"
set +e
python -m pip install --prefer-binary -r "$REQ_FILTERED"
batch_status=$?
set -e
if [[ "$batch_status" -ne 0 ]]; then
  log "批量安装失败，改为逐包安装（失败的包会跳过）"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" ]] && continue
    python -m pip install --prefer-binary $line && continue
    echo "WARN: skip $line"
  done < "$REQ_FILTERED"
fi
rm -f "$REQ_FILTERED"

# ONNX 语音 tokenizer：优先 GPU runtime，没有 cp313 轮子则退回 CPU
python - <<'PY'
import importlib.util
import subprocess
import sys

def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

if not installed("onnxruntime"):
    for pkg in ("onnxruntime-gpu", "onnxruntime"):
        print(f"try {pkg}", flush=True)
        if subprocess.call([sys.executable, "-m", "pip", "install", "--prefer-binary", pkg]) == 0:
            break
PY

python -m pip install --prefer-binary modelscope huggingface_hub gradio HyperPyYAML wetext WeTextProcessing inflect

log "安装 qwen-tts（--no-deps，避免把 transformers / huggingface_hub / gradio 降级）"
python -m pip install --prefer-binary --no-deps qwen-tts
python -m pip install --prefer-binary sox
# 若之前已被 qwen-tts 0.1.1 拉低 huggingface_hub，diffusers 会报冲突；拉回 1.x
python -m pip install --prefer-binary "huggingface_hub>=1.23"

if [[ ! -f "${MODEL_DIR}/llm.rl.pt" ]]; then
  log "下载 Fun-CosyVoice3-0.5B-2512（含 llm.rl.pt），约 7GB+"
  python "${ROOT}/download_models.py" --out "$MODEL_DIR" --source "$DOWNLOAD_SOURCE"
else
  log "已有模型权重，跳过下载: ${MODEL_DIR}"
fi

log "Colab 环境就绪。"
python - <<'PY'
import importlib.metadata as md
import sys
import torch

def ver(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "missing"

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", ver("transformers"), "huggingface_hub", ver("huggingface_hub"))
print("gradio", ver("gradio"), "qwen-tts", ver("qwen-tts"))
try:
    import qwen_tts  # noqa: F401
    print("qwen_tts import OK")
except Exception as exc:  # noqa: BLE001
    print("qwen_tts import FAIL", exc)
if torch.cuda.is_available():
    print(
        "gpu",
        torch.cuda.get_device_name(0),
        "vram_gb",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    )
PY
