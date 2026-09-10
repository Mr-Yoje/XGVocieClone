#!/usr/bin/env bash
# Ubuntu 一键启动：补齐环境与权重后打开 Web 对比界面。
#
# 用法:
#   bash start_ubuntu.sh
#   bash start_ubuntu.sh --listen          # 监听 0.0.0.0，便于局域网访问
#   bash start_ubuntu.sh --port 7860
#   bash start_ubuntu.sh --fp16 --force-gpu
#   bash start_ubuntu.sh --setup-only      # 只装环境，不下载、不启动
#   bash start_ubuntu.sh --skip-download   # 假设权重已下好

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosyvoice}"
MODEL_DIR="${ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"
SETUP_ONLY=0
SKIP_DOWNLOAD=0
SERVER_NAME="127.0.0.1"
WEBUI_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only)
      SETUP_ONLY=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --listen)
      SERVER_NAME="0.0.0.0"
      shift
      ;;
    --port|--server-name|--cosyvoice-root|--model-dir)
      WEBUI_ARGS+=("$1" "$2")
      if [[ "$1" == "--server-name" ]]; then
        SERVER_NAME="$2"
      fi
      shift 2
      ;;
    --fp16|--force-gpu|--share)
      WEBUI_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      WEBUI_ARGS+=("$1")
      shift
      ;;
  esac
done

bash "${ROOT}/setup_env.sh"

if [[ "$SETUP_ONLY" -eq 1 ]]; then
  echo "已按 --setup-only 结束。"
  exit 0
fi

find_conda_sh() {
  local candidates=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "${HOME}/miniforge3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "${ROOT}/.miniconda3/etc/profile.d/conda.sh"
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -f "$p" ]] && echo "$p" && return 0
  done
  if command -v conda >/dev/null; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]] && echo "${base}/etc/profile.d/conda.sh" && return 0
  fi
  return 1
}

CONDA_SH="$(find_conda_sh)"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

if [[ "$SKIP_DOWNLOAD" -ne 1 ]]; then
  if [[ ! -f "${MODEL_DIR}/llm.rl.pt" || ! -f "${MODEL_DIR}/cosyvoice3.yaml" ]]; then
    echo "==> 下载 Fun-CosyVoice3-0.5B-2512（含 llm.rl.pt）"
    python "${ROOT}/download_models.py" --out "$MODEL_DIR"
  else
    echo "==> 已检测到模型权重，跳过下载: ${MODEL_DIR}"
  fi
fi

echo "==> 启动 Web 界面  http://${SERVER_NAME}:7860  （Ctrl+C 结束）"
exec python "${ROOT}/webui_clone.py" --server-name "$SERVER_NAME" "${WEBUI_ARGS[@]}"
