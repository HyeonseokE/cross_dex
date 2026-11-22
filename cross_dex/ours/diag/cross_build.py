"""Cross-hand replay inputs for the space-vs-ceiling evaluation (RSS2027; dexlatent venv).

Hand s's teleop episode replayed on hand t (open loop, scene = s's episode):
  wrist    s's wrist actions -> t's wrist actions reaching the same canonical-frame wrist pose (dexverse_xlvla.wrist)
  fingers  mode dexpilot  DexPilot(s -> t) per frame (cross_dex.ours.uniform_pairs.retarget, the objective that made the
                          space's training pairs) = the kinematic ceiling of a space trained on those pairs
           mode latent    D_t(E_s(q_s)) of the given latent checkpoint (XL-VLA format)
           mode contact   oracle intent: DexPilot init, then wrist + fingers moved so t's fingertips reach s's WORLD
                          fingertip positions (object-frame contact targets; contact_retarget.py)
Writes <out>/actions.npz (per episode, t's action order) and <out>/demos.pkl (t's pickle with s's scene and t's start
= converted first step, written by the baseline's build_demos.py with the simulator env).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.diag.cross_build \
      --task <T> --src s --dst t --mode dexpilot|latent [--ckpt <pt>] --wrist <calib.json> --episodes 0-4 --out <dir>
"""
import argparse
import json
import os
import pickle
import subprocess

import numpy as np
import torch

from dexverse_xlvla import hands as H
from dexverse_xlvla.replay import MAIN_PYTHON, action_names, episode_list, pkl_path
from dexverse_xlvla.wrist import load_wrists

DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}   # uniform_pairs defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["dexpilot", "latent", "contact"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--episodes", default="0-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--starts", type=int, default=2)
    a = ap.parse_args()
    assert a.src != a.dst
    a.out = os.path.abspath(a.out); os.makedirs(a.out, exist_ok=True)
    eps = episode_list(a.episodes)
    src_pkl, dst_pkl = pkl_path(a.task, a.src), pkl_path(a.task, a.dst)
    an, bn = action_names(a.task, a.src), action_names(a.task, a.dst)
    src = pickle.load(open(src_pkl, "rb"))
    if a.mode == "latent":
        from dexverse_xlvla.eval_latent import dec, enc, load
        tr = load(a.ckpt)
        for m in tr.autoencoders.values():
            m.eval()
        models = tr.hand_models
    else:
        from cross_dex.ours.uniform_pairs import retarget
        models = {h: H.CanonicalHandModel(h) for h in (a.src, a.dst)}
    ma, mb = models[a.src], models[a.dst]
    ia, ib = ma.data_index(an), mb.data_index(bn)
    wr = load_wrists(a.wrist, {h: models[h] for h in (a.src, a.dst)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    actions, start, stats = {}, {}, {}
    for k in eps:
        act = np.asarray(src["episodes"][k]["actions"], dtype=np.float64)
        qa = ma.angles_to_normalized(torch.tensor(act[:, ia], dtype=torch.float32)).clamp(-1, 1)
        if a.mode == "latent":
            with torch.no_grad():
                qb = dec(tr, a.dst, enc(tr, a.src, qa)).clamp(-1, 1)
        else:
            q, rms, _ = retarget(ma, mb, qa.to(dev), a.iters, a.starts, 0.05, 1e-6, DP)
            qb = q.detach().cpu().clamp(-1, 1)
            stats[k] = {"res_mm_median": round(float(rms.median()), 2)}
        vec_t = wr[a.src].vec_from_actions(act[:, :6])                 # same canonical-frame pose as s
        if a.mode == "contact":
            from cross_dex.ours.diag.contact_retarget import compose_wrist, tip_match
            q, D, st = tip_match(ma, mb, qa.to(dev), qb.to(dev))
            qb = q.clamp(-1, 1)
            vec_t = compose_wrist(vec_t, D)
            stats[k].update(st)
        out = np.zeros((len(act), len(bn)))
        lo, hi = mb._lower, mb._upper
        out[:, ib] = ((qb + 1) * 0.5 * (hi - lo) + lo).numpy()
        wb = wr[a.dst]
        out[:, :6] = wb.actions_from_vec(vec_t, q_init=wb.joints(np.zeros(6)))
        with torch.no_grad():
            allb = mb._normalized_to_all_joint_angles(qb[:1])
        stats.setdefault(k, {})["finger_step_mean"] = round(float(np.abs(np.diff(out[:, 6:], axis=0)).mean()), 5)
        alias = {v: kk for kk, v in H.HANDS[a.dst].data_name.items()}
        j = {n: float(v) for n, v in zip(bn[:6], wb.joints(out[0, :6]))}
        j.update({alias.get(n, n): float(v[0]) for n, v in allb.items()})
        start[str(k)] = j
        actions[k] = out.astype(np.float32)
        print(a.src, "->", a.dst, "ep", k, stats[k], flush=True)
    np.savez(f"{a.out}/actions.npz", **{str(k): v for k, v in actions.items()})
    spec = {"src_pkl": src_pkl, "dst_pkl": dst_pkl, "out": f"{a.out}/demos.pkl", "src_hand": a.src, "mode": a.mode,
            "episodes": eps, "start": start}
    json.dump(spec, open(f"{a.out}/demos_spec.json", "w"))
    import dexverse_xlvla
    bd = os.path.join(os.path.dirname(dexverse_xlvla.__file__), "build_demos.py")
    subprocess.run([MAIN_PYTHON, bd, f"{a.out}/demos_spec.json"], check=True, env={"PATH": os.environ.get("PATH", "")})
    json.dump({"task": a.task, "src": a.src, "dst": a.dst, "mode": a.mode, "ckpt": a.ckpt, "episodes": eps,
               "stats": {str(k): v for k, v in stats.items()}}, open(f"{a.out}/build.json", "w"), indent=1)


if __name__ == "__main__":
    main()
