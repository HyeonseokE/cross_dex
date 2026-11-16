#!/usr/bin/env python
"""Run the common evaluator unchanged (DexVerse baselines/lerobot_dp/eval_lerobot.py) with our pi0.5 session server
(session_server.py, DexLatent venv) in place of its LeRobot policy server. Same mechanism as the baseline's
dexverse_xlvla/run_eval.py (which starts dexverse_xlvla/server.py); that file is not imported or changed.
Run in the main env (Isaac Sim), like the evaluator. Evaluator arguments pass through; ours:

  --mode shared|dedicated   shared: chunks from the GPU's openpi_server.py (--shared_port, $SHARED_EVAL_AUTHKEY)
                            dedicated: the baseline policy class spawns its own openpi process (old path, + timing)
  --barrier DIR:N  --timing <jsonl>  --save_obs K      see session_server.py

  source /workspace/codes/_setup/dexverse_env.sh DexVerse
  SHARED_EVAL_AUTHKEY=<k> python run_eval_shared.py --mode shared --shared_port <q> \
      --demos <pkl> --ckpt <cell folder with server.json> --output_dir <dir> --headless [--episodes 0,1]
"""
import argparse
import os
import runpy
import subprocess
import sys


def _server() -> str:  # the container's one /workspace/experiments/<server> folder (as dexverse_xlvla/run_eval.py)
    return next(d for d in sorted(os.listdir("/workspace/experiments"))
                if os.path.isdir(f"/workspace/experiments/{d}") and d[0] not in "_.")


own = argparse.ArgumentParser(add_help=False)
own.add_argument("--mode", choices=("shared", "dedicated"), default="shared")
own.add_argument("--shared_port", type=int, default=0)
own.add_argument("--barrier", default=None)
own.add_argument("--timing", default=None)
own.add_argument("--save_obs", type=int, default=0)
own.add_argument("--help_shared", action="store_true", help="print this wrapper's help and exit")
mine, rest = own.parse_known_args()
if mine.help_shared:
    print(__doc__)
    sys.exit(0)
if mine.mode == "shared" and not (mine.shared_port and os.environ.get("SHARED_EVAL_AUTHKEY")):
    sys.exit("--mode shared needs --shared_port and $SHARED_EVAL_AUTHKEY")

episodes = argparse.ArgumentParser(add_help=False)
episodes.add_argument("--episodes", default="")
EPISODES = episodes.parse_known_args(rest)[0].episodes

EVALUATOR = os.path.join(os.environ.get("XLVLA_COMMON_DP") or
                         f"/workspace/experiments/{_server()}/20261031_teleop_single/teleop_dp_image", "eval_lerobot.py")
PYTHON = os.environ.get("XLVLA_DEXLATENT_PYTHON", "/workspace/envs/dexverse/dexlatent/bin/python")
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_server.py")
EXTRA = ["--mode", mine.mode, "--shared_port", str(mine.shared_port), "--episodes", EPISODES]
for flag in ("barrier", "timing"):
    if getattr(mine, flag):
        EXTRA += [f"--{flag}", getattr(mine, flag)]
if mine.save_obs:
    EXTRA += ["--save_obs", str(mine.save_obs)]

_Popen = subprocess.Popen


class _SwapServer(_Popen):
    def __init__(self, cmd, *a, **k):
        if isinstance(cmd, list) and len(cmd) > 1 and str(cmd[1]).endswith("policy_server.py"):
            cmd = [PYTHON, SERVER] + list(cmd[2:]) + EXTRA
        super().__init__(cmd, *a, **k)


def swapped_command(cmd):
    """for the dry run / tests: the command the evaluator's Popen gets"""
    return [PYTHON, SERVER] + list(cmd[2:]) + EXTRA


if __name__ == "__main__":
    subprocess.Popen = _SwapServer
    sys.argv = [EVALUATOR] + rest
    runpy.run_path(EVALUATOR, run_name="__main__")
