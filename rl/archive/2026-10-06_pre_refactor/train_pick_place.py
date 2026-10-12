"""Train/play the pick-and-place env with Isaac Lab's rsl_rl scripts.

  ./isaaclab.sh -p ~/pan/6D_pose/rl/train_pick_place.py --num_envs 64              # GUI, small
  ./isaaclab.sh -p ~/pan/6D_pose/rl/train_pick_place.py --num_envs 2048 --headless
  ./isaaclab.sh -p ~/pan/6D_pose/rl/train_pick_place.py --play --num_envs 16 --checkpoint <model.pt>
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = Path(os.environ.get("ISAACLAB_PATH", Path.home() / "IsaacLab"))
sys.path.insert(0, str(HERE))
RL = LAB / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(RL))

play = "--play" in sys.argv
if play:
    sys.argv.remove("--play")
task = "Pick-Place-M1013-2F85-Play-v0" if play else "Pick-Place-M1013-2F85-v0"
if "--task" not in sys.argv:
    sys.argv += ["--task", task]

# Isaac Lab's scripts import task packages after the app starts; ours registers itself on import, which needs the app
# running too, so hook the import into gym registration via isaaclab_tasks' own import hook.
import isaaclab.app  # noqa: E402,F401  (ensures isaaclab is importable before the script launches the app)

_orig_import = __import__


def _install_registration():
    import builtins

    def hook(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if name == "isaaclab_tasks" and "pick_place_env" not in sys.modules:
            _orig_import("pick_place_env")
        return mod
    builtins.__import__ = hook


_install_registration()
runpy.run_path(str(RL / ("play.py" if play else "train.py")), run_name="__main__")
