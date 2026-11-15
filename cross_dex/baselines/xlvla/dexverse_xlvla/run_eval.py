#!/usr/bin/env python
"""Run the common evaluator unchanged, with the XL-VLA policy server in place of its LeRobot policy server.

The evaluator (experiments/<server>/20261031_teleop_single/teleop_dp_image/eval_lerobot.py) starts
``[<lerobot python>, <its dir>/policy_server.py, --ckpt ...]`` with subprocess.Popen; this wrapper swaps that one command
for ``[<dexlatent python>, dexverse_xlvla/server.py, --ckpt ...]`` and runs the evaluator as __main__ with the given
arguments. Nothing in the evaluator file changes. Run in the main env (Isaac Sim), like the evaluator itself:

  source /workspace/codes/_setup/dexverse_env.sh DexVerse-develop
  python run_eval.py --demos <pkl> --ckpt <folder with server.json> --output_dir <dir> --headless [--episodes 0,1]
"""
import os
import runpy
import subprocess
import sys

def _server() -> str:   # the container's one /workspace/experiments/<server> folder
    return next(d for d in sorted(os.listdir("/workspace/experiments"))
                if os.path.isdir(f"/workspace/experiments/{d}") and d[0] not in "_.")


EVALUATOR = os.path.join(os.environ.get("XLVLA_COMMON_DP") or
                         f"/workspace/experiments/{_server()}/20261031_teleop_single/teleop_dp_image", "eval_lerobot.py")
PYTHON = "/workspace/envs/dexverse/dexlatent/bin/python"
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

_Popen = subprocess.Popen


class _SwapServer(_Popen):
    def __init__(self, cmd, *a, **k):
        if isinstance(cmd, list) and len(cmd) > 1 and str(cmd[1]).endswith("policy_server.py"):
            cmd = [PYTHON, SERVER] + list(cmd[2:])
        super().__init__(cmd, *a, **k)


subprocess.Popen = _SwapServer
sys.argv = [EVALUATOR] + sys.argv[1:]
runpy.run_path(EVALUATOR, run_name="__main__")
