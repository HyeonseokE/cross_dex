"""Latent checks on one training checkpoint (started by train_latent.py --monitor_every, or by hand), CPU only.

  python -m dexverse_xlvla.latent_monitor --ckpt <checkpoint_epoch_N.pt> --run_dir <run> --tasks <t1,t2,...>

Check 1 (numbers): eval_latent metrics (paper Tab. 5 + the same on the tasks' teleop actions, pooled)
  -> <run>/artifacts/monitor/step_NNNNN.json, rows (step, eval, <hand>/<metric>, value) in <run>/artifacts/monitor/metrics.csv
Check 2 (picture): one latent decoded on every hand, drawn as link skeletons in the hand-common canonical frame
  -> <run>/artifacts/monitor/step_NNNNN.png. Rows: a thumb-index pinch of each hand, then one teleop frame per task
  (hand = row index mod hands, frame = the most closed hand among the first 2000 frames). The source hand's title is red;
  the other cells should show the same grasp.
Check 3 (simulator replay) is dexverse_xlvla/replay_monitor.sh.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import HandLatent.kinematics as K  # noqa: E402
from dexverse_xlvla.check_hands import dataset_dir, load_actions  # noqa: E402
from dexverse_xlvla.eval_latent import dec, enc, evaluate, load_eval, task_data  # noqa: E402

TIP_COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]   # thumb, index, middle, ring, little


def skeleton(model, q: torch.Tensor):
    """Link origins (canonical frame) and parent->child segments for normalized joints q (B, D); same FK as DexLatent's
    HandKinematicsModel._forward_internal, keeping every link instead of the fingertips only."""
    angles = model._normalized_to_all_joint_angles(q)
    b = q.shape[0]
    tf = {model.root_link: torch.eye(4).repeat(b, 1, 1)}
    segs = []
    for jn in model.traversal_order:
        s = model.joint_specs[jn]
        o = s.origin_transform.float().unsqueeze(0).repeat(b, 1, 1)
        j = torch.eye(4).repeat(b, 1, 1)
        if s.kind == "revolute":
            j[:, :3, :3] = K.axis_angle_to_matrix(s.axis.float().unsqueeze(0).repeat(b, 1), angles[jn].float())
        tf[s.child] = tf[s.parent] @ o @ j
        segs.append((s.parent, s.child))
    c = model.canon.float()
    pts = {k: (v[:, :3, 3] @ c[:3, :3].T + c[:3, 3]).numpy() for k, v in tf.items()}
    return pts, segs


def picture(tr, tasks, path: str, step: int) -> None:
    names = tr.hand_names
    rows = []                                                       # (label, source hand, normalized source joints)
    for h in names:
        rows.append((f"pinch 0-1 of {h}", h, tr._pinch_templates[h][:1]))
    for i, t in enumerate(tasks):
        h = names[i % len(names)]
        m = tr.hand_models[h]
        act, cols = load_actions(dataset_dir(t, h))
        qn = m.angles_to_normalized(torch.tensor(act[:2000, m.data_index(cols)], dtype=torch.float32)).clamp(-1, 1)
        k = int((qn - m.open_qpos()).norm(dim=-1).argmax())                 # most closed hand in the first frames
        rows.append((f"{t.replace('Dexverse-', '')} frame {k} of {h}", h, qn[k:k + 1]))
    fig = plt.figure(figsize=(3.2 * len(names), 3.0 * len(rows)))
    with torch.no_grad():
        for r, (label, src, q) in enumerate(rows):
            z = enc(tr, src, q)
            for c, h in enumerate(names):
                qh = q if h == src else dec(tr, h, z)
                pts, segs = skeleton(tr.hand_models[h], qh)
                tips = tr.hand_models[h].forward(qh)[0].numpy()
                ax = fig.add_subplot(len(rows), len(names), r * len(names) + c + 1, projection="3d")
                for a, b in segs:
                    ax.plot(*np.stack([pts[a][0], pts[b][0]]).T, color="0.4", lw=1)
                for i, p in enumerate(tips):
                    ax.scatter(*p, color=TIP_COLORS[i], s=18)
                allp = np.concatenate([v[0][None] for v in pts.values()] + [tips])
                ctr, half = allp.mean(0), 0.55 * np.ptp(allp, 0).max()             # zoom on the hand, equal axes
                ax.set_xlim(ctr[0] - half, ctr[0] + half), ax.set_ylim(ctr[1] - half, ctr[1] + half)
                ax.set_zlim(ctr[2] - half, ctr[2] + half)
                ax.view_init(elev=20, azim=-60)
                ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
                ax.set_title((f"{h}  [source]" if h == src else h), fontsize=9,
                             color="black" if h != src else "tab:red")
                if c == 0:
                    ax.text2D(-0.25, 0.5, label, transform=ax.transAxes, rotation=90, va="center", fontsize=8)
    fig.suptitle(f"step {step}: one latent decoded on every hand (canonical frame, tips: thumb red .. little purple)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated tasks (teleop data of every hand)")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    step = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"])
    d = os.path.join(args.run_dir, "artifacts", "monitor")
    os.makedirs(d, exist_ok=True)
    tr = load_eval(args.ckpt)
    out = evaluate(tr, task_data(tr, tasks))
    json.dump(out, open(os.path.join(d, f"step_{step:05d}.json"), "w"), indent=1)
    with open(os.path.join(d, "metrics.csv"), "a") as f:                     # monitors of several steps may overlap
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write("step,split,metric,value\n")
        for h, v in out.items():
            for k, x in v.items():
                f.write(f"{step},eval,{h}/{k},{x:.6g}\n")
    picture(tr, tasks, os.path.join(d, f"step_{step:05d}.png"), step)
    m = out["mean"]
    print(f"[monitor] step {step}: data_recon_joint {m['data_recon_joint_rmse']:.4f} data_line_dir {m['data_line_dir_err']:.2f} "
          f"pinch_dir {m['pinch_dir_err']:.2f} random_dir {m['random_dir_err']:.2f}", flush=True)


if __name__ == "__main__":
    main()
