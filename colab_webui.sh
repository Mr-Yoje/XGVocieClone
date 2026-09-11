#!/usr/bin/env bash
# Colab / 云端：前台启动 WebUI（日志直接打到当前单元格），以及停止。
# 用法（在仓库根目录）:
#   bash colab_webui.sh start    # 前台跑 python，实时输出；若已在跑则先停再启
#   bash colab_webui.sh stop
#   bash colab_webui.sh status
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "${ROOT}/outputs"

PID_FILE="${WEBUI_PID_FILE:-${ROOT}/outputs/webui.pid}"
PORT="${WEBUI_PORT:-7860}"

is_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  tr -d ' \n' <"$PID_FILE"
}

cmd_status() {
  local pid
  pid="$(read_pid || true)"
  if is_alive "${pid:-}"; then
    echo "WebUI 运行中 pid=${pid}"
    return 0
  fi
  echo "WebUI 未运行。"
  return 1
}

cmd_stop() {
  local pid
  pid="$(read_pid || true)"
  if is_alive "${pid:-}"; then
    echo "优雅停止 pid=${pid}（SIGINT）…"
    kill -INT "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
      is_alive "$pid" || break
      sleep 1
    done
    if is_alive "$pid"; then
      echo "仍未退出，发送 SIGTERM…"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 3
    fi
    if is_alive "$pid"; then
      echo "强制结束 SIGKILL…"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  pkill -f "python -u webui_clone.py" 2>/dev/null || true
  pkill -f "python webui_clone.py" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "WebUI 已停止。"
}

cmd_start() {
  local pid
  pid="$(read_pid || true)"
  if is_alive "${pid:-}"; then
    echo "已在运行 pid=${pid}，先停止再前台启动…"
    cmd_stop
  fi
  echo "前台启动 webui_clone.py，日志会实时打在本格。"
  echo $$ >"$PID_FILE"
  export PYTHONUNBUFFERED=1
  exec python -u webui_clone.py \
    --share --fp16 --force-gpu \
    --server-name 0.0.0.0 --port "$PORT"
}

usage() {
  sed -n '2,7p' "$0"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
