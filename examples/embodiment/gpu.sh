#!/usr/bin/env bash
set -euo pipefail

# ===================== 配置区：只改这里 =====================
GPU_INDEX=0                 # 监控哪张 GPU（0/1/2...）
THRESHOLD_MIB=2000          # 触发阈值：显存“已用” < 这个值（单位 MiB）
INTERVAL_SEC=300             # 轮询间隔（秒）

CMD_SCRIPT="/home/my/data/wxh/openpi/RLinf/examples/embodiment/pipe1.sh"  # 你要执行的 bash 脚本（必须是绝对路径）
RUN_ONCE=1                  # 1=触发一次后退出；0=持续监控，满足就可能触发
COOLDOWN_SEC=10            # RUN_ONCE=0 时生效：两次触发最小间隔（秒）

LOG_FILE="/tmp/gpu_watch_${GPU_INDEX}.log"      # 日志文件（可改/可留空）
# ============================================================

log() {
  local msg="[$(date '+%F %T')] $*"
  echo "$msg"
  if [[ -n "${LOG_FILE:-}" ]]; then
    echo "$msg" >> "$LOG_FILE"
  fi
}

# 基础检查
if [[ ! -f "$CMD_SCRIPT" ]]; then
  log "ERROR: CMD_SCRIPT not found: $CMD_SCRIPT"
  exit 1
fi
if [[ ! -x "$CMD_SCRIPT" ]]; then
  log "ERROR: CMD_SCRIPT not executable. Run: chmod +x \"$CMD_SCRIPT\""
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "ERROR: nvidia-smi not found. Install NVIDIA driver / tools first."
  exit 1
fi

# 防重复运行（同一张卡只允许一个监控实例）
LOCK="/tmp/gpu_watch_${GPU_INDEX}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "Another watcher is already running for GPU $GPU_INDEX (lock: $LOCK). Exit."
  exit 0
fi

get_used_mib() {
  nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

last_run_ts=0
log "START gpu-watch: GPU_INDEX=$GPU_INDEX THRESHOLD_MIB=$THRESHOLD_MIB INTERVAL_SEC=$INTERVAL_SEC RUN_ONCE=$RUN_ONCE CMD_SCRIPT=$CMD_SCRIPT"

while true; do
  used="$(get_used_mib || true)"
  if [[ -z "$used" ]]; then
    log "WARN: nvidia-smi query failed; retry in ${INTERVAL_SEC}s"
    sleep "$INTERVAL_SEC"
    continue
  fi
  log "POLL: used=${used}MiB (threshold=${THRESHOLD_MIB}MiB)"  
  if (( used < THRESHOLD_MIB )); then
    if (( RUN_ONCE == 1 )); then
      log "TRIGGER: used=${used}MiB < ${THRESHOLD_MIB}MiB => run once and exit"
      bash "$CMD_SCRIPT" >> "${LOG_FILE:-/dev/null}" 2>&1
      exit 0
    else
      now="$(date +%s)"
      if (( now - last_run_ts >= COOLDOWN_SEC )); then
        log "TRIGGER: used=${used}MiB < ${THRESHOLD_MIB}MiB => run (cooldown ok)"
        bash "$CMD_SCRIPT" >> "${LOG_FILE:-/dev/null}" 2>&1
        last_run_ts="$now"
      else
        log "SKIP: used=${used}MiB < ${THRESHOLD_MIB}MiB but cooldown (${COOLDOWN_SEC}s) not reached"
      fi
    fi
  fi

  sleep "$INTERVAL_SEC"
done
