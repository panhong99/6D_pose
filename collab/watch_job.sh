#!/usr/bin/env bash
# Agent watchdog (10/14): watches ONE long job and always exits within --max_min, printing why, so the agent that
# started it in the background is woken.  It never waits forever: done / died / stalled / time-up all end it.
#   bash collab/watch_job.sh --pid PID [--log FILE] [--done FILE] [--stall_min 20] [--max_min 30]
#     --pid    the job's process id (the job runs on its own, e.g. nohup ... & echo $!)
#     --log    file that grows while the job works (stall check)
#     --done   file the job writes when it has finished (e.g. summary.json)
# Exit line: WATCH <done|died|ended|stalled|timeup>  (ended = pid gone, log then quiet for 2 min) pid=... elapsed=...min log_age=...min  (+ last log lines)
set -uo pipefail
PID="" LOG="" DONE="" STALL=20 MAX=30
while [ $# -gt 0 ]; do
  case "$1" in
    --pid) PID="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --done) DONE="$2"; shift 2 ;;
    --stall_min) STALL="$2"; shift 2 ;;
    --max_min) MAX="$2"; shift 2 ;;
    *) echo "WATCH error: unknown argument $1"; exit 2 ;;
  esac
done
[ -n "$PID" ] || { echo "WATCH error: --pid is required"; exit 2; }
STALL_S=$(awk "BEGIN{print int($STALL*60)}"); MAX_S=$(awk "BEGIN{print int($MAX*60)}")   # minutes may be decimals
START_TICKS=$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || echo "")   # guards against a reused pid
t0=$(date +%s)
report() {
  local why="$1" now age="-"
  now=$(date +%s)
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then age=$(( (now - $(stat -c %Y "$LOG")) / 60 )); fi
  echo "WATCH $why pid=$PID elapsed=$(( (now - t0) / 60 ))min log_age=${age}min"
  if [ -n "$LOG" ] && [ -f "$LOG" ]; then tail -n 5 "$LOG" | cut -c1-200; fi
  exit 0
}
alive() {
  [ -n "$START_TICKS" ] && [ "$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null)" = "$START_TICKS" ]
}
while true; do
  [ -n "$DONE" ] && [ -f "$DONE" ] && report done
  if ! alive; then                                   # (10/14) a wrapper pid can end while its child keeps writing the log:
    if [ -n "$LOG" ] && [ -f "$LOG" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -lt 120 ]; then
      PID_GONE=1                                      # follow the log until it has been quiet for 2 min
    else
      [ -n "${PID_GONE:-}" ] && report ended || report died
    fi
  fi
  now=$(date +%s)
  if [ -n "$LOG" ] && [ -f "$LOG" ] && [ $(( now - $(stat -c %Y "$LOG") )) -ge "$STALL_S" ]; then report stalled; fi
  [ $(( now - t0 )) -ge "$MAX_S" ] && report timeup
  sleep 2
done
