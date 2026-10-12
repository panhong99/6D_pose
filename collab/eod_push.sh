#!/usr/bin/env bash
# End-of-day commit + push of CODE / CONFIG only (Human 2026-10-08: "퇴근" -> commit & push; never data or model weights).
#   bash collab/eod_push.sh --dry-run     list what would be committed, change nothing
#   bash collab/eod_push.sh ["message"]   commit on the current branch and push it to origin
# Staged: new or modified files with a whitelisted extension, outside the data / log / weight folders, <= 1 MB,
# plus deletions of tracked files.  Anything else (npz, pt, pth, onnx, png, jpg, mp4, bag, ply, obj, json, log, ...) stays out.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
MSG="${1:-EOD $(date '+%Y-%m-%d'): code and config}"

EXT='\.(py|sh|bash|yaml|yml|md|rviz|toml|cfg|ini|txt|urdf|xacro|launch|xml)$'
SKIP='(^|/)(data|logs|log|wandb|weights|checkpoints|outputs|runs|backups|reports|demo_data|debug|exp|\.runtime|__pycache__)(/|$)|^IL/data|^calibration/data/'
MAX_BYTES=1048576

mapfile -t candidates < <(git ls-files --modified --others --exclude-standard | grep -E "$EXT" | grep -vE "$SKIP" || true)
files=()
for f in "${candidates[@]}"; do
  [ -f "$f" ] || continue
  if [ "$(stat -c %s "$f")" -gt "$MAX_BYTES" ]; then echo "skip (>1 MB): $f"; continue; fi
  files+=("$f")
done
mapfile -t deleted < <(git ls-files --deleted || true)

# refuse obvious secrets
if [ "${#files[@]}" -gt 0 ] && grep -nIE '(api[_-]?key|secret|password|token)[[:space:]]*[:=][[:space:]]*["'"'"']?[A-Za-z0-9_\-]{16,}' "${files[@]}"; then
  echo "STOP: possible secret above; nothing committed"; exit 1
fi

echo "branch: $(git branch --show-current) -> origin"
echo "add/modify (${#files[@]}):"; printf '  %s\n' "${files[@]}"
echo "delete (${#deleted[@]}):"; printf '  %s\n' "${deleted[@]}"
if [ "$DRY" = 1 ]; then echo "(dry run: nothing committed)"; exit 0; fi
if [ "${#files[@]}" -eq 0 ] && [ "${#deleted[@]}" -eq 0 ]; then echo "nothing to commit"; exit 0; fi

git reset -q                                         # stage exactly the list below, nothing pre-staged
[ "${#files[@]}" -gt 0 ] && git add -- "${files[@]}"
[ "${#deleted[@]}" -gt 0 ] && git rm -q --cached -- "${deleted[@]}"
git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin "$(git branch --show-current)"
git log --oneline -1
