#!/usr/bin/env bash
# Colab / 云端：后台启动、查看日志、优雅停止 WebUI。
# 用法（在仓库根目录）:
#   bash colab_webui.sh start           # 后台跑，并把日志打到当前终端
#   bash colab_webui.sh start --no-follow
#   bash colab_webui.sh logs
#   bash colab_webui.sh follow
#   bash colab_webui.sh status
#   bash colab_webui.sh stop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "${ROOT}/outputs"

PID_FILE="${WEBUI_PID_FILE:-${ROOT}/outputs/webui.pid}"
LOG_FILE="${WEBUI_LOG_FILE:-${ROOT}/outputs/webui.log}"
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
    echo "WebUI 运行中 pid=${pid}  日志: ${LOG_FILE}"
    return 0
  fi
  echo "WebUI 未运行。"
  return 1
}

cmd_start() {
  local pid follow="${WEBUI_FOLLOW:-1}"
  if [[ "${2:-}" == "--no-follow" ]] || [[ "${1:-}" == "--no-follow" ]]; then
    follow=0
  fi
  pid="$(read_pid || true)"
  if is_alive "${pid:-}"; then
    echo "已在运行 pid=${pid}。下面跟随日志（中断单元格只停止显示，不会关 WebUI）。"
    if [[ "$follow" == "1" ]]; then
      cmd_follow
    else
      cmd_logs
    fi
    return 0
  fi
  rm -f "$PID_FILE"
  : >"$LOG_FILE"
  echo "启动 webui_clone.py，日志同时写入终端和 ${LOG_FILE}"
  # -u：print/logging 立刻出现在终端，不要等缓冲
  PYTHONUNBUFFERED=1 nohup python -u webui_clone.py \
    --share --fp16 --force-gpu \
    --server-name 0.0.0.0 --port "$PORT" \
    >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "pid=$(cat "$PID_FILE")"
  if [[ "$follow" == "1" ]]; then
    echo "正在把日志打到终端。中断本格只停止跟随；退出请运行 stop。"
    cmd_follow
  else
    sleep 8
    cmd_logs
  fi
}

cmd_follow() {
  local pid
  pid="$(read_pid || true)"
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "(还没有日志)"
    return 0
  fi
  trap 'echo; echo "已停止跟随日志。WebUI 若仍在运行，用 bash colab_webui.sh stop 退出。"; exit 0' INT TERM
  if is_alive "${pid:-}"; then
    tail -n +1 -f "$LOG_FILE" --pid="$pid"
  else
    tail -n +1 -f "$LOG_FILE"
  fi
}

cmd_logs() {
  echo "----- ${LOG_FILE}（末尾 80 行）-----"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 80 "$LOG_FILE"
  else
    echo "(还没有日志)"
  fi
}

cmd_stop() {
  local pid
  pid="$(read_pid || true)"
  if ! is_alive "${pid:-}"; then
    # 兜底：杀掉占用端口的进程
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    pkill -f "python webui_clone.py" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "没有在跑的 WebUI。"
    return 0
  fi
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
  pkill -f "python webui_clone.py" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "WebUI 已停止。"
}

usage() {
  sed -n '2,8p' "$0"
}

case "${1:-}" in
  start) cmd_start "${2:-}" ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  follow) cmd_follow ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
