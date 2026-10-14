#!/usr/bin/env bash
# Claude's Codex watcher (10/14 보스): ends as soon as Codex commits (a commit without Claude's co-author line) or the
# task board changes on disk, and in any case after --max_min, so Claude is woken to review and hand out the next task.
#   bash collab/watch_codex.sh [--max_min 25]
# Exit line: CODEX <commit|board|timeup> ... (+ the new commit subjects)
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MAX=25
[ "${1:-}" = "--max_min" ] && MAX="$2"
head0=$(git rev-parse HEAD)
board0=$(stat -c %Y collab/TASKBOARD.md)
t0=$(date +%s)
while true; do
  new=$(git log --format='%H' "$head0"..HEAD 2>/dev/null)
  for h in $new; do
    if ! git log -1 --format=%B "$h" | grep -q 'Co-Authored-By: Claude'; then
      echo "CODEX commit after $(( ($(date +%s) - t0) / 60 ))min:"
      git log --format='  %h %ad %s' --date=format:%H:%M "$head0"..HEAD | grep -v '^$'
      exit 0
    fi
  done
  if [ "$(stat -c %Y collab/TASKBOARD.md)" != "$board0" ] && ! git diff --quiet -- collab/TASKBOARD.md; then
    echo "CODEX board: collab/TASKBOARD.md changed (not committed yet)"; exit 0
  fi
  [ $(( $(date +%s) - t0 )) -ge "$(awk "BEGIN{print int($MAX*60)}")" ] && { echo "CODEX timeup ${MAX}min: no Codex commit"; exit 0; }
  sleep 15
done
