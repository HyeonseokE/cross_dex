#!/usr/bin/env python
"""Run the common evaluator (DexVerse baselines/lerobot_dp/eval_lerobot.py) unchanged, with this folder's server.py in
place of its LeRobot policy server (same swap as dexverse_xlvla/run_eval.py). Main env (Isaac Sim):

  python run_eval.py --demos <pkl> --ckpt <folder with server.json> --output_dir <dir> --episodes 0,1 --headless [--no-video]
"""
import os
import runpy
import subprocess
import sys

EVALUATOR = os.path.join(os.environ.get("COMMON_DP", "/workspace/codes/dexverse/DexVerse/baselines/lerobot_dp"),
                         "eval_lerobot.py")
PYTHON = "/workspace/envs/dexverse/dexlatent/bin/python"
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
_Popen = subprocess.Popen


class _SwapServer(_Popen):
    def __init__(self, cmd, *a, **k):
        if isinstance(cmd, list) and len(cmd) > 1 and str(cmd[1]).endswith("policy_server.py"):
            cmd = [PYTHON, SERVER] + list(cmd[2:])
            env = dict(k.get("env") or os.environ)
            env["PYTHONPATH"] = "/workspace/codes/dexverse/DexLatent:/workspace/codes/cross_dex"
            k["env"] = env
        super().__init__(cmd, *a, **k)


subprocess.Popen = _SwapServer
sys.argv = [EVALUATOR] + sys.argv[1:]
runpy.run_path(EVALUATOR, run_name="__main__")
