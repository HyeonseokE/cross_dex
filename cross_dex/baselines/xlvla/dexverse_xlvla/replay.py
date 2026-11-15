"""Step 4: latent replay (paper Tab. 4) on DexVerse. Builds what the unchanged common evaluator
(experiments/<server>/20261031_teleop_single/teleop_dp_image/eval_lerobot.py, through run_eval.py) needs to replay hand A's
demonstrations on hand B:

  actions.npz   per episode, B's action sequence (B action order):
                  fingers  = B joints decoded from A's latent  (mode latent)  or A's own joints (mode raw, A == B)
                  wrist    = B wrist actions reaching A's canonical-frame wrist pose (dexverse_xlvla.wrist), A != B
  demos.pkl     (A != B) B's pickle with every episode's objects / goal taken from A's episode and B's starting wrist /
                fingers set to the converted first step (B's own start would not match A's object layout)
  policy/       descriptor for dexverse_xlvla.server (mode replay)

  python -m dexverse_xlvla.replay --src shadow --dst allegro --mode latent --ckpt <latent.pt> --wrist <calib.json>
         --out <dir> [--episodes 0-19]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess

import numpy as np
import torch

from dexverse_xlvla import hands as H
from dexverse_xlvla.check_hands import dataset_dir
from dexverse_xlvla.eval_latent import dec, enc, load
from dexverse_xlvla.wrist import load_wrists

MAIN_PYTHON = "/workspace/envs/dexverse/main/bin/python"   # simulator env (called, not changed)
PKL_ROOT = os.environ.get("XLVLA_PKL_ROOT", "/workspace/datasets/dexverse/teleop_pkl_bs1")


def pkl_path(task: str, hand: str) -> str:
    hits = sorted(glob.glob(f"{PKL_ROOT}/*/{task}/{task}_*_floating_{hand}_right.pkl"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{task} {hand}: {hits}")
    return hits[0]


def action_names(task: str, hand: str) -> list:
    info = json.load(open(f"{dataset_dir(task, hand)}/meta/info.json"))
    n = info["features"]["action"]["names"]
    return list(n.values())[0] if isinstance(n, dict) else n


def episode_list(spec: str, n: int = 50) -> list:
    if not spec:
        return list(range(n))
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["raw", "latent"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--wrist")
    ap.add_argument("--episodes", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    if args.mode == "raw" and args.src != args.dst:
        raise ValueError("raw replay is only defined for src == dst")
    os.makedirs(f"{args.out}/policy", exist_ok=True)
    eps = episode_list(args.episodes)

    src_pkl, dst_pkl = pkl_path(args.task, args.src), pkl_path(args.task, args.dst)
    a_names, b_names = action_names(args.task, args.src), action_names(args.task, args.dst)
    src = pickle.load(open(src_pkl, "rb"))
    tr = load(args.ckpt) if args.mode == "latent" else None
    models = tr.hand_models if tr else {h: H.CanonicalHandModel(h) for h in {args.src, args.dst}}
    ma, mb = models[args.src], models[args.dst]
    ia, ib = ma.data_index(a_names), mb.data_index(b_names)
    cross = args.src != args.dst
    wrists = load_wrists(args.wrist, {h: models[h] for h in (args.src, args.dst)}) if cross else None

    actions, firsts = {}, {}
    for k in eps:
        act = np.asarray(src["episodes"][k]["actions"], dtype=np.float64)          # A action order
        out = np.zeros((len(act), len(b_names)))
        if args.mode == "raw":
            out[:] = act
        else:
            with torch.no_grad():
                za = enc(tr, args.src, ma.angles_to_normalized(torch.tensor(act[:, ia], dtype=torch.float32)).clamp(-1, 1))
                qb = dec(tr, args.dst, za)
                lo, hi = mb._lower, mb._upper
                out[:, ib] = ((qb.clamp(-1, 1) + 1) * 0.5 * (hi - lo) + lo).numpy()
                allb = mb._normalized_to_all_joint_angles(qb[:1])                     # incl. mimic joints (start)
            if cross:
                wb = wrists[args.dst]
                out[:, :6] = wb.actions_from_vec(wrists[args.src].vec_from_actions(act[:, :6]), q_init=wb.joints(np.zeros(6)))
                start = wb.joints(out[0, :6])                                         # B joints = A's first target
            else:
                out[:, :6] = act[:, :6]
                start = out[0, :6]                                                    # unused (self replay)
            firsts[k] = {"wrist": start, "fingers": {j: float(v[0]) for j, v in allb.items()}}
        actions[k] = out.astype(np.float32)
    np.savez(f"{args.out}/actions.npz", **{str(k): v for k, v in actions.items()})

    demos = src_pkl
    if cross:   # B's pickle with A's scene per episode; written by the simulator env (numpy 1.x, see build_demos.py)
        alias = {v: k for k, v in H.HANDS[args.dst].data_name.items()}             # URDF -> sim name
        start = {}
        for k in eps:
            j = {n: float(v) for n, v in zip(b_names[:6], firsts[k]["wrist"])}
            j.update({alias.get(n, n): v for n, v in firsts[k]["fingers"].items()})
            start[str(k)] = j
        demos = f"{args.out}/demos.pkl"
        spec = {"src_pkl": src_pkl, "dst_pkl": dst_pkl, "out": demos, "src_hand": args.src, "mode": args.mode,
                "episodes": eps, "start": start}
        json.dump(spec, open(f"{args.out}/demos_spec.json", "w"))
        subprocess.run([MAIN_PYTHON, os.path.join(os.path.dirname(__file__), "build_demos.py"),
                        f"{args.out}/demos_spec.json"], check=True, env={"PATH": os.environ.get("PATH", "")})
    json.dump({"mode": "replay", "actions": f"{args.out}/actions.npz", "episodes": eps},
              open(f"{args.out}/policy/server.json", "w"))
    # open-loop replay ends with the demo; the evaluator still waits up to max_steps for success -> demo length + 120
    json.dump({"src": args.src, "dst": args.dst, "mode": args.mode, "ckpt": args.ckpt, "episodes": eps,
               "max_steps": int(max(len(v) for v in actions.values())) + 120,
               "src_pkl": src_pkl, "demos": demos}, open(f"{args.out}/replay.json", "w"), indent=1)
    print(demos)


if __name__ == "__main__":
    main()
