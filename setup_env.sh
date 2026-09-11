#!/usr/bin/env bash
# Ubuntu: install CosyVoice runtime under this project.
# Requires: sudo (for sox/ffmpeg) and network. GPU driver optional.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosyvoice}"
COSY_DIR="${ROOT}/third_party/CosyVoice"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_HOST="${PIP_HOST:-mirrors.aliyun.com}"

log() { printf '\n==> %s\n' "$*"; }

ensure_apt() {
  local pkgs=()
  command -v git >/dev/null || pkgs+=(git)
  command -v wget >/dev/null || pkgs+=(wget)
  command -v curl >/dev/null || pkgs+=(curl)
  command -v sox >/dev/null || pkgs+=(sox libsox-dev)
  command -v ffmpeg >/dev/null || pkgs+=(ffmpeg)
  if ((${#pkgs[@]} == 0)); then
    return 0
  fi
  if ! command -v sudo >/dev/null; then
    echo "缺少系统依赖: ${pkgs[*]}。请先安装 sudo/apt 后重试。"
    exit 1
  fi
  log "安装系统依赖: ${pkgs[*]}"
  sudo apt-get update
  sudo apt-get install -y "${pkgs[@]}"
}

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
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  if command -v conda >/dev/null; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

install_miniconda() {
  local prefix="${ROOT}/.miniconda3"
  if [[ -f "${prefix}/etc/profile.d/conda.sh" ]]; then
    echo "${prefix}/etc/profile.d/conda.sh"
    return 0
  fi
  log "未检测到 conda，安装 Miniconda 到 ${prefix}"
  local installer="/tmp/Miniconda3-latest-Linux-x86_64.sh"
  wget -q -O "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$installer" -b -p "$prefix"
  echo "${prefix}/etc/profile.d/conda.sh"
}

ensure_cosyvoice_repo() {
  if [[ ! -d "$COSY_DIR/.git" ]]; then
    mkdir -p "${ROOT}/third_party"
    log "克隆 FunAudioLLM/CosyVoice（含子模块）"
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSY_DIR"
  else
    log "CosyVoice 已存在: ${COSY_DIR}"
    git -C "$COSY_DIR" submodule update --init --recursive
  fi
}

ensure_apt
ensure_cosyvoice_repo

CONDA_SH="$(find_conda_sh || true)"
if [[ -z "${CONDA_SH}" ]]; then
  CONDA_SH="$(install_miniconda)"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

log "创建/更新 conda 环境 ${CONDA_ENV_NAME} (python=3.10)"
conda create -n "$CONDA_ENV_NAME" -y python=3.10
conda activate "$CONDA_ENV_NAME"
python -m pip install -U pip
python -m pip install -r "${COSY_DIR}/requirements.txt" -i "$PIP_INDEX" --trusted-host="$PIP_HOST"
python -m pip install modelscope huggingface_hub gradio torchaudio qwen-tts -i "$PIP_INDEX" --trusted-host="$PIP_HOST"

log "环境已就绪。"
echo "  conda 初始化文件: ${CONDA_SH}"
echo "  环境名: ${CONDA_ENV_NAME}"
echo "下一步："
echo "  source ${CONDA_SH} && conda activate ${CONDA_ENV_NAME}"
echo "  python download_models.py"
echo "  python webui_clone.py"
echo "或直接: bash start_ubuntu.sh"
