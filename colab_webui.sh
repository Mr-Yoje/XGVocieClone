#!/usr/bin/env bash
# Colab：停掉占用 7860 的 WebUI 子进程。
# 用法（仓库根目录）: bash colab_webui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${WEBUI_PORT:-7860}"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
pkill -f "python -u webui_clone.py" 2>/dev/null || true
pkill -f "python webui_clone.py" 2>/dev/null || true
rm -f "${ROOT}/outputs/webui.pid"
echo "WebUI 已停止。"
