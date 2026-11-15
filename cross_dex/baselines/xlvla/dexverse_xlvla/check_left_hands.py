"""Checks before training the left-hand (bimanual) XL-VLA latent (CLAUDE.md: verify action semantics first).

Per left hand:
  1. the hand-only URDF builds, the fingertips exist, every model DoF is found in the bimanual dataset action names
     (dataset -> URDF name map in hands.LEFT_HANDS);
  2. the recorded left-hand finger actions of a bimanual demo lie inside the left URDF joint limits and move (not constant);
  3. mirror check: the same joint vector on the left and on the right hand gives the same canonical fingertips up to the
     left/right mirror (right canonical tips vs left tips with y flipped, compared by pairwise fingertip distances,
     which no mirror changes) -> catches a joint order / sign mismatch between the left and right models.

  python -m dexverse_xlvla.check_left_hands [--out left_check.json]
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

from dexverse_xlvla import hands as H

ROOT = "/workspace/datasets/dexverse/teleop_lerobot"


def bimanual_actions(hand: str):
    """(names, actions [T, D]) of the first bimanual episode of this hand, or None."""
    infos = sorted(glob.glob(f"{ROOT}/*/{hand}_bimanual/*/meta/info.json"))
    if not infos:
        return None
    info = json.load(open(infos[0]))
    names = info["features"]["action"]["names"]
    if isinstance(names, dict):
        names = list(names.values())[0]
    pq = sorted(glob.glob(infos[0].replace("meta/info.json", "data/*/*.parquet")))[0]
    df = pd.read_parquet(pq)
    ep = df[df["episode_index"] == df["episode_index"].min()]
    return names, np.stack(ep["action"].to_numpy()), infos[0].split("/")[-5]


def pdist(t):  # [N, m, 3]
    return torch.cdist(t, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--hands", default=",".join(H.LEFT_HAND_NAMES), help="left hands (name ends with _left)")
    a = ap.parse_args()
    rep = {}
    for left in a.hands.split(","):
        right = left[:-5]
        r = {}
        mL, mR = H.CanonicalHandModel(left), H.CanonicalHandModel(right)
        r["dof"] = [mL.dof_count(), mR.dof_count()]
        r["same_joint_names"] = [j for j in mL.dof_joints] == [j for j in mR.dof_joints] or \
            [j.replace("left_", "") for j in mL.dof_joints] == [j.replace("right_", "") for j in mR.dof_joints]
        got = bimanual_actions(right)
        if got is None:
            r["dataset"] = "no bimanual dataset"
        else:
            names, act, task = got
            idx = mL.data_index(names)
            q = torch.tensor(act[:, idx], dtype=torch.float32)
            lo, hi = mL._lower.cpu(), mL._upper.cpu()
            r["dataset"] = task
            r["frac_inside_limits"] = round(float(((q >= lo - 1e-3) & (q <= hi + 1e-3)).float().mean()), 4)
            r["joint_std_min_max"] = [round(float(q.std(0).min()), 4), round(float(q.std(0).max()), 4)]
        # mirror check on random joint vectors (normalized [-1, 1], same on both hands)
        g = torch.Generator().manual_seed(0)
        z = torch.rand(256, mL.dof_count(), generator=g) * 2 - 1
        tl, _, _ = mL._forward_internal(z)
        tr, _, _ = mR._forward_internal(z)
        r["mirror_tipdist_err_mm"] = round(float((pdist(tl) - pdist(tr)).abs().mean()) * 1e3, 2)
        perm = torch.randperm(256, generator=g)
        r["unrelated_pose_err_mm"] = round(float((pdist(tl) - pdist(tr[perm])).abs().mean()) * 1e3, 2)
        rep[left] = r
        print(left, json.dumps(r), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
