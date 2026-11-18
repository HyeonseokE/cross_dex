"""Hand-detail loss diagnosis, step 1 (openpi venv): open-loop pi0.5 predictions on demo frames.

Same frames are fed to a latent-space policy and a joint-space policy that share data, prompts and training recipe
(XL-VLA pi05 multitask: pi05-vae6-mt-goal vs pi05-joint-mt-goal). The latent and joint LeRobot sets are built from the
same demos in the same episode order (checked: wrist 9 dims of state/action agree), so frame i is the same moment in both.
Frames: per (task, hand) cell, episodes spread over the 50 and frames spread over each episode.

  python -m cross_dex.ours.diag.detail_infer --policy lat:<ckpt>:<run>:<repo_id>:41:<latent set>/all \
      --policy joint:<ckpt>:<run>:<repo_id>:32:<joint set>/all --out <npz> [--eps_per_cell 10 --frames_per_ep 20]
Writes <out>: frame index, cell, per policy: state, gt action chunk (H, D), predicted chunk (H, D).
"""
import argparse, json, os, pathlib, sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name:ckpt:run:repo_id:action_dim:dataset_root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps_per_cell", type=int, default=10)
    ap.add_argument("--frames_per_ep", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10)
    a = ap.parse_args()
    sys.path.insert(0, os.environ.get("OPENPI_ROOT", "/workspace/codes/dexverse/openpi"))
    from openpi.policies import policy_config
    from dexverse_xlvla.pi05.openpi_train import make_config
    from dexverse_xlvla.pi05.v3_dataset import V3Dataset

    specs = [p.split(":") for p in a.policy]
    dss = {s[0]: V3Dataset(s[5], a.horizon) for s in specs}
    ref = dss[specs[0][0]]
    cells = json.load(open(f"{specs[0][5]}/meta/xlvla_cells.json"))["cells"]
    n_ep = len(ref.eps); per = n_ep // len(cells)
    assert per * len(cells) == n_ep, (n_ep, len(cells))
    for name, ds in dss.items():                                   # same frames in every set
        assert len(ds) == len(ref) and np.allclose(ds.state[:, :9], ref.state[:, :9], atol=1e-4), name
    idx, cell_of = [], []
    for c in range(len(cells)):
        for e in np.linspace(c * per, (c + 1) * per - 1, a.eps_per_cell).round().astype(int):
            lo, hi = int(ref.eps.loc[e, "dataset_from_index"]), int(ref.eps.loc[e, "dataset_to_index"])
            for i in np.linspace(lo, hi - 1, a.frames_per_ep).round().astype(int):
                idx.append(i); cell_of.append(c)
    idx, cell_of = np.array(idx), np.array(cell_of)
    out = {"index": idx, "cell": cell_of, "cells": np.array(json.dumps(cells))}
    for name, ckpt, run, repo, adim, root in specs:
        cfg = argparse.Namespace(run=run, exp="eval", action_dim=int(adim), horizon=a.horizon, repo_id=repo, max_token_len=None,
                                 batch=1, accum=1, workers=0, warmup=1, lr=1e-4, steps=1, decay_lr=1e-6, ema=0, seed=42,
                                 log_interval=1, save_interval=1, overwrite=False, resume=False, fsdp=1)
        policy = policy_config.create_trained_policy(make_config(cfg), pathlib.Path(ckpt))
        ds = dss[name]; pred, gt, st = [], [], []
        for k, i in enumerate(idx):
            x = ds[int(i)]
            pred.append(np.asarray(policy.infer({"image": x["image"], "wrist_image": x["wrist_image"], "state": x["state"],
                                                 "prompt": x["prompt"]})["actions"], np.float32))
            gt.append(x["actions"]); st.append(x["state"])
            if k % 200 == 0:
                print(f"[{name}] {k}/{len(idx)}", flush=True)
        out[f"{name}_pred"], out[f"{name}_gt"], out[f"{name}_state"] = np.stack(pred), np.stack(gt), np.stack(st)
        del policy
        np.savez(a.out, **out)
        print(f"[{name}] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
