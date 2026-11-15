"""Step 3: evaluate a trained DexVerse hand latent with the paper's Tab. 5 metrics, plus the same metrics on the
teleop data distribution (GraspKettle actions), which the paper does not report.

  python -m dexverse_xlvla.eval_latent --run_dir <run> --ckpt <checkpoint.pt> [--task Dexverse-GraspKettle-v0]

Metrics (paper Sec. 4.2 definitions; lower is better), per hand and averaged:
  recon_joint_rmse (rad), recon_tip_rmse (m)            encode -> decode on the same hand
  pinch_dir_err (deg), pinch_dist_err (m)               pinch poses of a source hand decoded on every other hand:
                                                        angle / |length| difference of the thumb->finger lines
  random_dir_err (deg), random_dist_err (m)             same with random poses
  continuity_joint (rad), continuity_tip (m)            decode(z + N(0, 0.05^2)) vs decode(z)
  interp_accel, interp_jerk (m / step^k)                fingertip trajectory of decoded linear latent interpolation
Data metrics (prefix data_): recon and cross-hand thumb-finger line errors on the hand's teleop actions
(--task a,b,...: the tasks' actions pooled).
Appends rows (split = eval) to <run>/metrics.csv and writes <run>/artifacts/latent_eval.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

import HandLatent.model as M
from dexverse_xlvla import hands as H
from dexverse_xlvla.check_hands import dataset_dir, load_actions
from dexverse_xlvla.train_latent import DexVerseTrainer, SampledHandAutoencoder

sys.path.insert(0, "/workspace/experiments")
from seed_control import seed_everything  # noqa: E402

N_RANDOM, N_PINCH, N_INTERP, INTERP_STEPS, EPS = 4096, 1024, 256, 32, 0.05


def load(ckpt: str):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = payload["config"]
    H.install()
    if c.get("lambda_kl", 0) > 0:          # paper variant
        M.HandAutoencoder = SampledHandAutoencoder
    cfg = M.TrainingConfig(arm_dof=0, latent_dim_hand=c["latent_dim_hand"], hand_hidden_dims=tuple(c["hand_hidden_dims"]),
                           pinch_template_count=N_PINCH, checkpoint_dir="/workspace/.tmp/xlvla_eval_unused",
                           device=torch.device("cpu"))     # inference on CPU (small MLPs; data and FK are on CPU)
    tr = DexVerseTrainer(payload["hand_names"], cfg)
    tr.load_autoencoders_from_payload(payload)
    return tr


def enc(tr, h, q):
    return tr.autoencoders[h].encode(q)[1]          # posterior mean (deterministic)


def dec(tr, h, z):
    return tr.autoencoders[h].decode_from_latents(z[:, :0], z)[1]


def lines(tips, pairs):
    return torch.stack([tips[:, 0] - tips[:, k] for _, k in pairs], 1)       # (B, P, 3) thumb - finger


def line_err(src_tips, dst_tips, pairs):
    a, b = lines(src_tips, pairs), lines(dst_tips, pairs)
    cos = (torch.nn.functional.normalize(a, dim=-1) * torch.nn.functional.normalize(b, dim=-1)).sum(-1).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cos)).mean()), float((a.norm(dim=-1) - b.norm(dim=-1)).abs().mean())


@torch.no_grad()
def cross(tr, src_q: dict, prefix: str, out: dict):
    names = tr.hand_names
    per = {}
    for s in names:
        z = enc(tr, s, src_q[s])
        st = tr.hand_models[s].forward(src_q[s])
        d_, l_ = [], []
        for t in names:
            if t == s:
                continue
            pairs = tr.shared_pinch_pairs(s, t)
            tt = tr.hand_models[t].forward(dec(tr, t, z))
            d, l = line_err(st, tt, pairs)
            d_.append(d)
            l_.append(l)
        per[s] = (float(np.mean(d_)), float(np.mean(l_)))
        out.setdefault(s, {})[f"{prefix}_dir_err"] = per[s][0]
        out[s][f"{prefix}_dist_err"] = per[s][1]


@torch.no_grad()
def recon(tr, h, q, prefix, out):
    qh = dec(tr, h, enc(tr, h, q))
    m = tr.hand_models[h]
    lo, hi = m._lower, m._upper
    ang = lambda x: (x.clamp(-1, 1) + 1) * 0.5 * (hi - lo) + lo                   # normalized -> rad
    out.setdefault(h, {})[f"{prefix}recon_joint_rmse"] = float(((ang(qh) - ang(q)) ** 2).mean().sqrt())
    out[h][f"{prefix}recon_tip_rmse"] = float(((m.forward(qh) - m.forward(q)) ** 2).sum(-1).mean().sqrt())


def load_eval(ckpt: str):
    """Trainer from a checkpoint with the evaluation pinch poses (cached next to the checkpoint; they depend only on the
    hand models, so checkpoints of one run share the cache)."""
    seed_everything(1000)                       # evaluation pinch poses (fixed)
    tr = load(ckpt)
    cache =os.path.join(os.path.dirname(ckpt), "pinch_templates_eval.pt")   # pinch poses for the pinch metric
    if os.path.exists(cache):
        tr._pinch_templates = torch.load(cache)
    else:
        tr._cache_pinch_templates()
        torch.save(tr._pinch_templates, f"{cache}.{os.getpid()}")              # monitors may build it at the same time
        os.replace(f"{cache}.{os.getpid()}", cache)
    for a in tr.autoencoders.values():
        a.eval()
    return tr


def task_data(tr, tasks) -> dict:
    """Normalized finger joints of each hand's teleop actions, all given tasks concatenated."""
    data = {}
    for h in tr.hand_names:
        m, qs = tr.hand_models[h], []
        for t in tasks:
            act, cols = load_actions(dataset_dir(t, h))
            qs.append(torch.tensor(act[:, m.data_index(cols)], dtype=torch.float32))
        data[h] = m.angles_to_normalized(torch.cat(qs)).clamp(-1, 1)
    return data


def evaluate(tr, data: dict) -> dict:
    """All metrics (module docstring) per hand and their mean; fixed evaluation samples (seed 1000)."""
    seed_everything(1000)                       # evaluation samples (fixed, independent of the training seed)
    out: dict = {}
    names = tr.hand_names
    rnd = {h: torch.empty(N_RANDOM, tr.hand_models[h].dof_count()).uniform_(-1, 1) for h in names}
    pinch = {h: tr._pinch_templates[h][torch.randperm(len(tr._pinch_templates[h]))[:N_PINCH]] for h in names}
    with torch.no_grad():
        for h in names:
            recon(tr, h, rnd[h], "", out)
            recon(tr, h, data[h], "data_", out)
            m = tr.hand_models[h]
            z = enc(tr, h, rnd[h])
            q0, q1 = dec(tr, h, z), dec(tr, h, z + EPS * torch.randn_like(z))
            lo, hi = m._lower, m._upper
            out[h]["continuity_joint"] = float(((q1 - q0) * 0.5 * (hi - lo)).norm(dim=-1).mean())
            out[h]["continuity_tip"] = float((m.forward(q1) - m.forward(q0)).norm(dim=-1).mean())
            za, zb = enc(tr, h, rnd[h][:N_INTERP]), enc(tr, h, rnd[h][N_INTERP:2 * N_INTERP])
            w = torch.linspace(0, 1, INTERP_STEPS).view(-1, 1, 1)
            zt = (1 - w) * za + w * zb                                                # (S, B, L)
            tips = m.forward(dec(tr, h, zt.reshape(-1, zt.shape[-1]))).reshape(INTERP_STEPS, N_INTERP, -1, 3)
            acc = tips[2:] - 2 * tips[1:-1] + tips[:-2]
            jerk = acc[1:] - acc[:-1]
            out[h]["interp_accel"] = float(acc.norm(dim=-1).mean())
            out[h]["interp_jerk"] = float(jerk.norm(dim=-1).mean())
    cross(tr, pinch, "pinch", out)
    cross(tr, rnd, "random", out)
    cross(tr, data, "data_line", out)
    keys = list(next(iter(out.values())).keys())
    out["mean"] = {k: float(np.mean([out[h][k] for h in names])) for k in keys}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0", help="task, or comma-separated tasks (data pooled)")
    args = ap.parse_args()
    tr = load_eval(args.ckpt)
    out = evaluate(tr, task_data(tr, args.task.split(",")))
    keys = list(out["mean"].keys())
    os.makedirs(os.path.join(args.run_dir, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(args.run_dir, "artifacts", "latent_eval.json"), "w"), indent=1)
    step = torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]
    with open(os.path.join(args.run_dir, "metrics.csv"), "a") as f:
        for h, d in out.items():
            for k, v in d.items():
                f.write(f"{step},eval,{h}/{k},{v:.6g}\n")
    print(f"{'hand':8s} " + " ".join(f"{k[:16]:>16s}" for k in keys))
    for h, d in out.items():
        print(f"{h:8s} " + " ".join(f"{d[k]:16.4f}" for k in keys))


if __name__ == "__main__":
    main()
