"""Functional-correction framework, step 1: per-step training arrays from the teleop demos (RSS2027, 2026-11-23).

For every (task, hand) teleop pickle: per step t (action t, state t)
  wrist    (9)          canonical wrist pose in the world (pos m + rot6d)   dexverse_xlvla.wrist (wrist_calib_<task>.json)
  q        (dof_h)      normalized finger command [-1, 1]                    CanonicalHandModel order
  k        (d_k)        shared-space finger latent E_h(q)                    --ckpt (stage 1, frozen)
  aw       (6)          raw wrist action                                     pkl actions[:, :6]
  obj_rel  (N_o, 9)     object pose in the canonical wrist frame
  obj_w    (N_o, 9)     object pose in the world (pos + rot6d)
  ep_start (n_ep + 1)   row offsets of the episodes
-> <out>/<task>/<hand>.npz + <hand>.json (object names, action names, dof, ckpt, quaternion order).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.fc.make_data \
      --ckpt <stage-1 model.pt> --wrist_dir <wrist_calib dir> --out /workspace/datasets/dexverse/functional_correction/v0
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from cross_dex.ours.diag.delta_knn import HANDS, SKIP, TASKS, quat_order, quat_to_R
from dexverse_xlvla.eval_latent import enc, load
from dexverse_xlvla.replay import action_names, pkl_path
from dexverse_xlvla.wrist import from_rot6d, load_wrists, rot6d


def one(task, h, tr, wrist_dir, out):
    try:
        p = pkl_path(task, h)
    except FileNotFoundError:
        return None
    d = pickle.load(open(p, "rb"))
    m = tr.hand_models[h]
    names = action_names(task, h)
    ia = m.data_index(names)
    wr = load_wrists(f"{wrist_dir}/wrist_calib_{task}.json", {h: m})[h]
    st0 = d["episodes"][0]["states"]
    objs = sorted(o for o in st0[0]["rigid_object"] if not o.startswith(SKIP))
    order = quat_order(st0)
    cols = {k: [] for k in ("wrist", "q", "k", "aw", "obj_rel", "obj_w")}
    starts, succ = [0], []
    for epi in d["episodes"]:
        act, st = np.asarray(epi["actions"], dtype=np.float64), epi["states"]
        T = min(len(act), len(st))
        act = act[:T]
        vec = wr.vec_from_actions(act[:, :6])
        Rw = from_rot6d(vec[:, 3:9])
        rel, wld = [], []
        for o in objs:
            P = np.stack([np.asarray(st[i]["rigid_object"][o]["root_pose"]).reshape(-1) for i in range(T)])
            Ro = quat_to_R(P[:, 3:7], order)
            rel.append(np.concatenate([np.einsum("nji,nj->ni", Rw, P[:, :3] - vec[:, :3]),
                                       rot6d(np.einsum("nji,njk->nik", Rw, Ro))], -1))
            wld.append(np.concatenate([P[:, :3], rot6d(Ro)], -1))
        q = m.angles_to_normalized(torch.tensor(act[:, ia], dtype=torch.float32)).clamp(-1, 1)
        with torch.no_grad():
            k = enc(tr, h, q).numpy()
        for key, v in (("wrist", vec), ("q", q.numpy()), ("k", k), ("aw", act[:, :6]),
                       ("obj_rel", np.stack(rel, 1)), ("obj_w", np.stack(wld, 1))):
            cols[key].append(np.asarray(v, dtype=np.float32))
        starts.append(starts[-1] + T)
        succ.append(bool(epi.get("success", True)))
    os.makedirs(f"{out}/{task}", exist_ok=True)
    np.savez_compressed(f"{out}/{task}/{h}.npz", ep_start=np.array(starts), success=np.array(succ),
                        **{k: np.concatenate(v) for k, v in cols.items()})
    meta = {"task": task, "hand": h, "pkl": p, "objects": objs, "quat_order": order, "action_names": names,
            "dof": int(cols["q"][0].shape[1]), "latent_dim": int(cols["k"][0].shape[1]), "episodes": len(succ),
            "steps": int(starts[-1])}
    json.dump(meta, open(f"{out}/{task}/{h}.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wrist_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    a = ap.parse_args()
    torch.set_num_threads(8)
    tr = load(a.ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    rows = []
    for task in a.tasks:
        for h in HANDS:
            meta = one(task, h, tr, a.wrist_dir, a.out)
            if meta:
                rows.append(meta)
                print(task, h, meta["episodes"], meta["steps"], "dof", meta["dof"], "objs", meta["objects"], flush=True)
            else:
                print(task, h, "missing", flush=True)
    json.dump({"ckpt": a.ckpt, "wrist_dir": a.wrist_dir, "sets": rows}, open(f"{a.out}/index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
