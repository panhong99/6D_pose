#!/usr/bin/env bash
# One arm leg with a FRESH plan (plans expire after 30 min) - 보스 runs it; the mover asks Enter before moving.
#   bash tesollo/go.sh plate|bowl LEG [deg/s=20] [--dx M --dy M]       legs of tesollo/arm_plan.py: <obj>_above, _down, _lift, _home
#   bash tesollo/go.sh basket-plate|basket-bowl LEG [deg/s=20] [basket_plan.py args]   legs: lift, to_above, descend, ascend, to_home
set -e
cd "$(dirname "$0")/.."
what=$1; leg=$2; vel=${3:-20}; shift 3 2>/dev/null || shift $#
PY=~/miniconda3/envs/foundationpose/bin/python
case $what in
  plate|bowl)  P=$($PY tesollo/arm_plan.py "$what" "$@" | awk '/^plan:/{print $2}') ;;
  basket-*)    P=$($PY tesollo/basket_plan.py --object "${what#basket-}" "$@" | awk '/^plan:/{print $2}') ;;
  *) echo "usage: bash tesollo/go.sh plate|bowl|basket-plate|basket-bowl LEG [deg/s] [planner args]"; exit 2 ;;
esac
[ -n "$P" ] || { echo "no plan (see the planner output above)"; exit 1; }
echo "plan $P"
exec bash calibration/run.sh mover --plan "$P" --leg "$leg" --smooth --go_per leg --vel_deg_s "$vel" --acc_deg_s2 "$vel" --execute --go_enter
