"""Assemble the cross-embodiment paired chunk dataset (pipeline step 7) from retargeting runs + their self-collision
reports.

Per trajectory (all hands of one retargeting run = the same human motion): sliding windows of H steps (stride S) of
every hand's finger targets ``<hand>_q`` (env action finger order = what a policy outputs). Quality filter (plan A,
2026-11-12): a trajectory is clean if no hand penetrates deeper than the report's pen_mm anywhere; a chunk is clean if
no hand does inside its window. Per (source, Feix type): if the type has >= --min_clean_eps clean trajectories, only
clean trajectories are used (episode-level filter); otherwise (types whose grasp closure collides on some hand) clean
chunks of all trajectories are used (chunk-level filter).

Output <out>/chunks.npz: q_<hand> [N,H,n_h] float32, feix [N], source [N] (index into sources), traj [N] (index into
trajs.json), t0 [N], err_<hand> [N] (mean DexPilot residual in the window, m), split [N] (0 train / 1 val, by
trajectory); <out>/meta.json (hands, joint names, H, stride, fps, per-type counts, normalisation stats from train).

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.build_chunks \
      --runs dexcanvas=<rt dir>,hograspnet=<rt dir> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from collections import Counter, defaultdict

import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest
from cross_dex.dataset.chunk_paired.select_trajs import DEXCANVAS_FEIX


def feix_of(z) -> int | None:
    if "feix" in z.files:
        return int(z["feix"])
    lab = str(z["action_label"])
    return DEXCANVAS_FEIX.get(int(lab.split("-")[0])) if lab[:1].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list source=<retarget dir> (each with <dir>_selfcollision.json)")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "chunks", "feix400_H16"))
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min_clean_eps", type=int, default=100,
                    help="per (source, type): episode-level filter if at least this many clean trajectories, else chunk-level")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--drop_pen_mm", type=float, default=8.0,
                    help="drop a whole trajectory if any hand's finger-finger penetration (after the collision correction) "
                         "exceeds this anywhere (user rule 2026-11-12: 8 mm); 0 = off")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [r.split("=", 1) for r in a.runs.split(",")]
    hands, names, fps = None, None, None
    n_drop: dict[str, int] = {}
    recs = []   # per trajectory: (source idx, path, feix, clean, pen frame sets per hand, T)
    for si, (src, rt) in enumerate(runs):
        meta = json.load(open(os.path.join(rt, "meta.json")))
        rep = json.load(open(rt.rstrip("/") + "_selfcollision.json"))
        pr, stride_c = rep["per_traj"], rep["args"].get("stride", 1)
        h = list(meta["hands"])
        if hands is None:
            hands, fps = h, meta["fps"]
            names = {k: meta["hands"][k]["finger_joint_names"] for k in hands}
        assert h == hands and meta["fps"] == fps, f"{rt}: hands / fps differ from the first run"
        for f in sorted(glob.glob(os.path.join(rt, "*.npz"))):
            tid = os.path.basename(f)[:-4]
            if tid not in pr:
                continue
            r = pr[tid]
            pen = {k: np.asarray(r[k]["pen_frames"], int) for k in hands}
            if a.drop_pen_mm and max(r[k]["max_pen_mm"] for k in hands) > a.drop_pen_mm:
                n_drop[src] = n_drop.get(src, 0) + 1
                continue
            recs.append({"src": si, "path": f, "tid": f"{src}/{tid}", "T": r["T"], "pen": pen, "stride_c": stride_c,
                         "clean": all(len(pen[k]) == 0 for k in hands)})
    # feix per trajectory (needs the npz; label only)
    for r in recs:
        with np.load(r["path"]) as z:
            r["feix"] = feix_of(z)
    recs = [r for r in recs if r["feix"] is not None]
    n_clean = Counter((r["src"], r["feix"]) for r in recs if r["clean"])
    n_all = Counter((r["src"], r["feix"]) for r in recs)
    mode = {k: ("episode" if n_clean[k] >= a.min_clean_eps else "chunk") for k in n_all}
    Q = {k: [] for k in hands}
    E = {k: [] for k in hands}
    feix, source, traj, t0s, split = [], [], [], [], []
    trajs = []
    for r in recs:
        m = mode[(r["src"], r["feix"])]
        if m == "episode" and not r["clean"]:
            continue
        z = np.load(r["path"])
        T = len(z["t"])
        ti = len(trajs)
        trajs.append(r["tid"])
        sp = int(zlib.crc32(r["tid"].encode()) % 10000 < a.val_frac * 10000)
        for t0 in range(0, T - a.H + 1, a.stride):
            if m == "chunk":
                # a checked frame f covers [f, f + stride_c) (self_collision checks every stride_c-th frame)
                if any(((pf + r["stride_c"] > t0) & (pf < t0 + a.H)).any() for pf in r["pen"].values()):
                    continue
            for k in hands:
                Q[k].append(z[f"{k}_q"][t0:t0 + a.H])
                E[k].append(float(z[f"{k}_err"][t0:t0 + a.H].mean()))
            feix.append(r["feix"]); source.append(r["src"]); traj.append(ti); t0s.append(t0); split.append(sp)
    out = {f"q_{k}": np.asarray(Q[k], np.float32) for k in hands}
    out.update({f"err_{k}": np.asarray(E[k], np.float32) for k in hands})
    out.update(feix=np.asarray(feix, np.int16), source=np.asarray(source, np.int8), traj=np.asarray(traj, np.int32),
               t0=np.asarray(t0s, np.int32), split=np.asarray(split, np.int8))
    np.savez(os.path.join(a.out, "chunks.npz"), **out)
    tr = out["split"] == 0
    norm = {k: {"mean": out[f"q_{k}"][tr].reshape(-1, len(names[k])).mean(0).tolist(),
                "std": out[f"q_{k}"][tr].reshape(-1, len(names[k])).std(0).tolist(),
                "min": out[f"q_{k}"][tr].reshape(-1, len(names[k])).min(0).tolist(),
                "max": out[f"q_{k}"][tr].reshape(-1, len(names[k])).max(0).tolist()} for k in hands}
    per = defaultdict(dict)
    fx, src = out["feix"], out["source"]
    for (si, g), n in sorted(n_all.items()):
        sel = (fx == g) & (src == si)
        per[runs[si][0]][int(g)] = {"trajs": n, "clean_trajs": n_clean[(si, g)], "filter": mode[(si, g)],
                                    "chunks": int(sel.sum()), "chunks_val": int((sel & (out["split"] == 1)).sum())}
    meta = {"hands": hands, "finger_joint_names": names, "H": a.H, "stride": a.stride, "fps": fps,
            "sources": [s for s, _ in runs], "runs": dict(runs), "n_chunks": int(len(fx)),
            "n_train": int(tr.sum()), "n_val": int((~tr).sum()), "per_source_feix": per, "norm": norm,
            "feix_types": sorted({int(g) for g in fx}), "dropped_deep_pen": n_drop, "args": vars(a)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    json.dump(trajs, open(os.path.join(a.out, "trajs.json"), "w"))
    save_manifest(a.out, a, n_chunks=meta["n_chunks"], n_train=meta["n_train"], n_val=meta["n_val"])
    print("dropped (penetration >", a.drop_pen_mm, "mm):", n_drop)
    print(json.dumps({"n": meta["n_chunks"], "train": meta["n_train"], "val": meta["n_val"], "types": meta["feix_types"]}))
    for s, d in per.items():
        for g, v in d.items():
            print(s, g, v)


if __name__ == "__main__":
    main()
