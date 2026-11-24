"""Action ceiling of each policy action space: a hand's own demos converted to the policy vector and back, written as
a trajectory pickle for the conversion replay path (record_demos --replay_demos, through pickle_to_lerobot).

  arm raw    recorded actions (no conversion)
  arm joint  wrist: action -> canonical pose 9 -> IK (warm start = previous solution, as the policy server);
             fingers: normalise -> clamp [-1, 1] -> back                       (= the joint DP's vector)
  arm <space> wrist as joint; fingers: D_h(E_h(q*))                              (= the latent DP's vector)

Two steps because the pickles are numpy 1.x (simulator env) and the encoders live in the dexlatent env:
  dexlatent: python -m cross_dex.ours.crosshand_dp.replay_build actions --hand h --arm a [--ckpt ..] --wrist .. --eps 0-9 --out x.npz
  main     : python replay_build.py pickle <src.pkl> <x.npz> <out.pkl>
"""
import glob
import os
import sys

import numpy as np

TASK = "Dexverse-GraspKettle-v0"
PKL_ROOT = os.environ.get("XLVLA_PKL_ROOT", "/workspace/datasets/dexverse/teleop_pkl_bs1")


def src_pkl(hand):
    hits = sorted(glob.glob(f"{PKL_ROOT}/*/{TASK}/{TASK}_*_floating_{hand}_right.pkl"))
    assert len(hits) == 1, hits
    return hits[0]


def eps_of(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in spec.split(",")]


def actions():
    import argparse
    import pickle

    import torch
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.eval_latent import dec, enc, load
    from dexverse_xlvla.replay import action_names
    from dexverse_xlvla.wrist import load_wrists

    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--eps", default="0-9")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(sys.argv[2:])
    h = a.hand
    tr = load(a.ckpt) if a.arm != "raw" else None   # joint: hand models of the checkpoint (as make_data)
    m = tr.hand_models[h] if tr else H.CanonicalHandModel(h)
    names = action_names(TASK, h)
    idx = m.data_index(names)
    w = load_wrists(a.wrist, {h: m})[h]
    src = pickle.load(open(src_pkl(h), "rb"))
    out, log = {}, []
    for k in eps_of(a.eps):
        act = np.asarray(src["episodes"][k]["actions"], dtype=np.float64)
        new = act.copy()
        if a.arm != "raw":
            vec = w.vec_from_actions(act[:, :6])
            new[:, :6] = w.actions_from_vec(vec, q_init=w.joints(act[0, :6]))
            qn = m.angles_to_normalized(torch.tensor(act[:, idx], dtype=torch.float32)).clamp(-1, 1)
            if a.arm != "joint":
                with torch.no_grad():
                    qn = dec(tr, h, enc(tr, h, qn)).clamp(-1, 1)
            lo, hi = m._lower, m._upper
            new[:, idx] = ((qn + 1) * 0.5 * (hi - lo) + lo).numpy()
        d = np.abs(new - act)
        log.append(f"ep {k}: wrist max |d| {d[:, :6].max():.2e}, finger mean |d| {d[:, 6:].mean():.4f} rad")
        out[str(k)] = new.astype(np.float32)
    np.savez(a.out, **out)
    print("\n".join(log))


def make_pickle():
    import pickle
    src, npz, dst = sys.argv[2:5]
    d = pickle.load(open(src, "rb"))
    z = np.load(npz)
    eps = sorted(int(k) for k in z.files)
    new = []
    for k in eps:
        e = dict(d["episodes"][k])
        old = np.asarray(e["actions"])
        assert z[str(k)].shape == old.shape, (z[str(k)].shape, old.shape)
        e["actions"] = z[str(k)].astype(old.dtype)
        new.append(e)
    d = dict(d)
    d["episodes"], d["num_episodes"] = new, len(new)
    d["replay_source"] = {"pickle": src, "actions": npz, "episodes": eps}
    pickle.dump(d, open(dst, "wb"))
    print(dst, len(new), "episodes")


if __name__ == "__main__":
    {"actions": actions, "pickle": make_pickle}[sys.argv[1]]()
