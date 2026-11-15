"""Adopted latent-space metrics (experiments/a4/RSS2027/README.md "공유 잠재 공간 평가 메트릭", 2026-11-15) for the
DexJoCo-X XL-VLA hand latents, per checkpoint. CPU only.

  1  demo self reconstruction     demo finger targets -> encode -> decode on the same hand: joint error (deg), fingertip error (mm)
  2  cross reconstruction         retargeting pair rows (held-out rows of the LAD split): latent of hand A decoded on hand B vs
                                  hand B's retargeted joints: joint error (deg), fingertip error (mm), mean over all A != B
  3  cross geometric consistency  demo frames: thumb-finger fingertip line of hand A vs of hand B decoded from A's latent:
                                  direction (deg), length (mm), mean over all A != B              (eval_latent.cross)
  (4, cross-hand latent replay in the simulator, is not available for DexJoCo-X yet.)

Demo data: training_v1/bundle/episodes/<task>/<hand>/scene_*/native.npz, action [T, 2 (right, left), 6 + n_act]; the left row
exists in the bimanual tasks only. Joint values outside the URDF limits are clipped by the model (fraction reported).

  python -m dexverse_xlvla.eval_space_dexjoco --ckpt_dir <dir with checkpoint_epoch_*.pt> --side right|left \
         --pairs <dexjoco_val20k.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from dexverse_xlvla import hands_dexjoco as HD
from dexverse_xlvla.eval_latent import cross, dec, enc, load_eval

DEMO_ROOT = os.environ.get("DEXJOCO_DEMO_ROOT", "/workspace/datasets/dexjoco/training_v1/bundle/episodes")


def demo_frames(hand: str, side: str, n_max: int, rng) -> tuple[np.ndarray, list[str]]:
    """Finger actuator targets [N, n_act] of every demo of this hand and side (random n_max frames), and the tasks used."""
    row = 0 if side == "right" else 1
    qs, tasks = [], set()
    for f in sorted(glob.glob(f"{DEMO_ROOT}/*/{HD.FILE[hand]}/scene_*/native.npz")):
        z = np.load(f)
        n = int(z["action_mask"][0, row].sum()) - 6
        if n <= 0:
            continue                                               # single-hand task: no left hand
        qs.append(z["action"][:, row, 6:6 + n].astype(np.float32))
        tasks.add(f.split("/")[-4])
    q = np.concatenate(qs)
    if len(q) > n_max:
        q = q[np.sort(rng.choice(len(q), n_max, replace=False))]
    return q, sorted(tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--side", choices=["right", "left"], required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_demo", type=int, default=30000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.makedirs(a.out, exist_ok=True)
    names = json.load(open(os.path.join(HD.G, "dataset_joint_names.json")))
    P = np.load(a.pairs, allow_pickle=True)
    pair_names = json.loads(str(P["names"]))
    rows = [f"{'step':>6} {'1 joint':>8} {'1 tip':>7} {'2 joint':>8} {'2 tip':>7} {'3 dir':>7} {'3 len':>7}"]
    data = pair = None
    summary = {}
    for ck in sorted(glob.glob(f"{a.ckpt_dir}/checkpoint_epoch_*.pt"), key=lambda s: int(s.split("_")[-1][:-3])):
        step = int(ck.split("_")[-1][:-3])
        tr = load_eval(ck)
        base = {h: h[3:].removesuffix("_left") for h in tr.hand_names}       # dj_<hand>[_left] -> <hand>
        if data is None:                                                       # same frames for every checkpoint
            rng = np.random.default_rng(1000)
            data, pair, info = {}, {}, {}
            for h in tr.hand_names:
                m = tr.hand_models[h]
                q, tasks = demo_frames(base[h], a.side, a.n_demo, rng)
                raw = m.angles_to_normalized(torch.tensor(q[:, m.data_index(names[base[h]])]))
                p = torch.tensor(P[f"{a.side}_{base[h]}"][:, m.data_index(pair_names[base[h]])])
                praw = m.angles_to_normalized(p)
                data[h], pair[h] = raw.clamp(-1, 1), praw.clamp(-1, 1)
                info[h] = {"demo_frames": len(q), "demo_tasks": tasks, "demo_frac_clipped": float((raw.abs() > 1 + 1e-4).float().mean()),
                           "pair_rows": len(p), "pair_frac_clipped": float((praw.abs() > 1 + 1e-4).float().mean())}
                print(h, json.dumps(info[h]), flush=True)
        out = {h: {} for h in tr.hand_names}
        deg = lambda m, d: torch.rad2deg(d.abs() * 0.5 * (m._upper - m._lower))       # normalised difference -> deg
        with torch.no_grad():
            for h in tr.hand_names:                                                    # 1
                m = tr.hand_models[h]
                qh = dec(tr, h, enc(tr, h, data[h])).clamp(-1, 1)
                out[h]["m1_demo_self_joint_deg"] = float(deg(m, qh - data[h]).mean())
                out[h]["m1_demo_self_tip_mm"] = float((m.forward(qh) - m.forward(data[h])).norm(dim=-1).mean() * 1e3)
            for s in tr.hand_names:                                                    # 2 (and the same on s == t)
                z = enc(tr, s, pair[s])
                j, t_ = [], []
                for t in tr.hand_names:
                    m = tr.hand_models[t]
                    qt = dec(tr, t, z).clamp(-1, 1)
                    ej = float(deg(m, qt - pair[t]).mean())
                    et = float((m.forward(qt) - m.forward(pair[t])).norm(dim=-1).mean() * 1e3)
                    if s == t:
                        out[s]["pair_self_joint_deg"], out[s]["pair_self_tip_mm"] = ej, et
                    else:
                        j.append(ej), t_.append(et)
                out[s]["m2_cross_joint_deg"], out[s]["m2_cross_tip_mm"] = float(np.mean(j)), float(np.mean(t_))
        cross(tr, data, "m3_demo_line", out)                                           # 3
        for h in tr.hand_names:
            out[h]["m3_demo_line_len_mm"] = out[h].pop("m3_demo_line_dist_err") * 1e3
            out[h]["m3_demo_line_dir_deg"] = out[h].pop("m3_demo_line_dir_err")
        keys = list(next(iter(out.values())))
        out["mean"] = {k: float(np.mean([out[h][k] for h in tr.hand_names])) for k in keys}
        out["_data"] = info
        json.dump(out, open(f"{a.out}/space_step_{step:05d}.json", "w"), indent=1)
        mm = out["mean"]
        summary[step] = mm
        rows.append(f"{step:>6} {mm['m1_demo_self_joint_deg']:>8.3f} {mm['m1_demo_self_tip_mm']:>7.2f} {mm['m2_cross_joint_deg']:>8.3f} "
                    f"{mm['m2_cross_tip_mm']:>7.2f} {mm['m3_demo_line_dir_deg']:>7.2f} {mm['m3_demo_line_len_mm']:>7.2f}")
        print(rows[0] + "\n" + rows[-1], flush=True)
    json.dump({"side": a.side, "mean_by_step": summary, "data": info}, open(f"{a.out}/space_summary.json", "w"), indent=1)
    open(f"{a.out}/space_summary.txt", "w").write("\n".join(rows) + "\n")
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
