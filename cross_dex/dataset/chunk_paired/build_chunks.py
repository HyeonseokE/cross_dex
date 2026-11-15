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

Per benchmark (2026-11-13): --bench dexverse|dexjoco picks the hand set, the output rate and the joint order (BENCH below);
--stretch lo,hi slows each human trajectory by a factor drawn in [lo, hi] (seeded by trajectory id) before resampling to
the benchmark rate, so finger speeds match the task demos (measure_speed.py: human median ~0.33 rad/s, demos 0.10-0.24).
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


# benchmark: hands (output name -> retarget hand), fps, joint order (index into the retarget hand's finger joints)
BENCH = {
    "dexverse": {"fps": 60, "hands": {h: h for h in ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]}, "order": {}},
    # DexJoCo-X actuator order (evaluation/contracts/<task>/<hand>.json actuator_names); allegro / leap: DexVerse finger
    # order already equals the actuator order (abduction, MCP, PIP, DIP per finger; checked by joint ranges 2026-11-13)
    "dexjoco": {"fps": 20, "hands": {"allegro": "allegro", "inspire": "inspire", "leap": "leap", "sharpa": "sharpa",
                                     "wuji": "wuji_dj", "linkerhand": "linkerhand", "xhand": "xhand"},
                "order": {"inspire": [4, 5, 0, 1, 3, 2],
                          "sharpa": [17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 8, 9, 10, 11, 12]}},
}


def resample(x, t_src, t_new):
    """linear interpolation of x [T, ...] given at t_src onto t_new"""
    i = np.clip(np.searchsorted(t_src, t_new, side="right") - 1, 0, len(t_src) - 2)
    w = np.clip((t_new - t_src[i]) / (t_src[i + 1] - t_src[i]), 0, 1).reshape(-1, *([1] * (x.ndim - 1)))
    return (x[i] * (1 - w) + x[i + 1] * w).astype(x.dtype), i


def feix_of(z) -> int | None:
    if "feix" in z.files:
        return int(z["feix"])
    lab = str(z["action_label"])
    return DEXCANVAS_FEIX.get(int(lab.split("-")[0])) if lab[:1].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list source=<retarget dir>[+<dir with more hands>] (each with <dir>_selfcollision.json)")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "chunks", "feix400_H16"))
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min_clean_eps", type=int, default=100,
                    help="per (source, type): episode-level filter if at least this many clean trajectories, else chunk-level")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--drop_pen_mm", type=float, default=8.0,
                    help="drop a whole trajectory if any hand's finger-finger penetration (after the collision correction) "
                         "exceeds this anywhere (user rule 2026-11-12: 8 mm); 0 = off")
    ap.add_argument("--bench", default="", choices=["", "dexverse", "dexjoco"],
                    help="benchmark set: hands, output fps and joint order (default: all hands at the source rate)")
    ap.add_argument("--stretch", default="1,1", help="lo,hi: time-stretch factor per trajectory (slower human motion)")
    ap.add_argument("--n_stretch", type=int, default=1, help="stretch draws per trajectory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s_lo, s_hi = map(float, a.stretch.split(","))
    bench_rt = set(BENCH[a.bench]["hands"].values()) if a.bench else None  # penetration filters use these hands only
    runs = [r.split("=", 1) for r in a.runs.split(",")]
    hands, names, fps = None, None, None
    n_drop: dict[str, int] = {}
    recs = []   # per trajectory: (source idx, path, feix, clean, pen frame sets per hand, T)
    for si, (src, rts) in enumerate(runs):
        # one source may combine several retargeting runs of the same trajectories with different hands
        # (e.g. <dir 4 hands>+<dir allegro,leap>); trajectories are joined by id, a trajectory is kept only if every run has it
        parts = []
        for rt in rts.split("+"):
            meta = json.load(open(os.path.join(rt, "meta.json")))
            rep = json.load(open(rt.rstrip("/") + "_selfcollision.json"))
            parts.append((rt, meta, rep["per_traj"], rep["args"].get("stride", 1)))
        h = [k for _, meta, _, _ in parts for k in meta["hands"]]
        fps_s = {meta["fps"] for _, meta, _, _ in parts}
        if hands is None:
            hands, fps = h, parts[0][1]["fps"]
            names = {k: meta["hands"][k]["finger_joint_names"] for _, meta, _, _ in parts for k in meta["hands"]}
        assert h == hands and fps_s == {fps}, f"{rts}: hands / fps differ from the first source ({h} vs {hands})"
        assert len({p[3] for p in parts}) == 1, "different self-collision strides"
        stride_c = parts[0][3]
        rt0 = parts[0][0]
        for f in sorted(glob.glob(os.path.join(rt0, "*.npz"))):
            tid = os.path.basename(f)[:-4]
            if not all(tid in pr for _, _, pr, _ in parts):
                continue
            path, pen, mx = {}, {}, {}
            for rt, meta, pr, _ in parts:
                for k in meta["hands"]:
                    if bench_rt is not None and k not in bench_rt:
                        continue
                    path[k] = os.path.join(rt, f"{tid}.npz")
                    pen[k] = np.asarray(pr[tid][k]["pen_frames"], int)
                    mx[k] = pr[tid][k]["max_pen_mm"]
            if a.drop_pen_mm and max(mx.values()) > a.drop_pen_mm:
                n_drop[src] = n_drop.get(src, 0) + 1
                continue
            recs.append({"src": si, "path": f, "paths": path, "tid": f"{src}/{tid}", "T": parts[0][2][tid]["T"], "pen": pen,
                         "stride_c": stride_c, "clean": all(len(v) == 0 for v in pen.values())})
    # feix per trajectory (needs the npz; label only)
    for r in recs:
        with np.load(r["path"]) as z:
            r["feix"] = feix_of(z)
    recs = [r for r in recs if r["feix"] is not None]
    n_clean = Counter((r["src"], r["feix"]) for r in recs if r["clean"])
    n_all = Counter((r["src"], r["feix"]) for r in recs)
    mode = {k: ("episode" if n_clean[k] >= a.min_clean_eps else "chunk") for k in n_all}
    fps_in = fps
    if a.bench:
        B = BENCH[a.bench]
        miss = [r for r in B["hands"].values() if r not in hands]
        assert not miss, f"runs lack hands {miss}"
        hmap, fps = B["hands"], B["fps"]
        names = {k: [names[r][j] for j in B["order"].get(k, range(len(names[r])))] for k, r in hmap.items()}
        order = {k: np.asarray(B["order"].get(k, range(len(names[k])))) for k in hmap}
    else:
        hmap, order = {k: k for k in hands}, {k: None for k in hands}
    pen_hands = list(hmap.values())
    out_hands = list(hmap)
    Q = {k: [] for k in out_hands}
    E = {k: [] for k in out_hands}
    stretch = []
    feix, source, traj, t0s, split = [], [], [], [], []
    trajs = []
    for r in recs:
        m = mode[(r["src"], r["feix"])]
        if m == "episode" and not r["clean"]:
            continue
        pens = {k: r["pen"][k] for k in pen_hands}
        zs = {p: np.load(p) for p in set(r["paths"].values())}
        z = zs[r["path"]]
        T = len(z["t"])
        t_src = np.arange(T) / fps_in
        sp = int(zlib.crc32(r["tid"].encode()) % 10000 < a.val_frac * 10000)
        rng = np.random.default_rng(zlib.crc32(r["tid"].encode()))
        qs = {k: zs[r["paths"][hmap[k]]][f"{hmap[k]}_q"] for k in out_hands}
        es = {k: zs[r["paths"][hmap[k]]][f"{hmap[k]}_err"].mean(-1) if zs[r["paths"][hmap[k]]][f"{hmap[k]}_err"].ndim > 1
              else zs[r["paths"][hmap[k]]][f"{hmap[k]}_err"] for k in out_hands}
        # covered source frames with penetration (a checked frame f covers [f, f + stride_c))
        bad = np.zeros(T, bool)
        for pf in pens.values():
            for f0 in pf:
                bad[f0:f0 + r["stride_c"]] = True
        for _ in range(a.n_stretch):
            s = float(rng.uniform(s_lo, s_hi)) if s_hi > s_lo else s_lo
            if s == 1.0 and fps == fps_in:
                rq, src_i = qs, np.arange(T)
                re = es
            else:
                t_new = np.arange(0, t_src[-1] * s + 1e-9, 1.0 / fps) / s
                rq, re = {}, {}
                for k in out_hands:
                    rq[k], src_i = resample(qs[k], t_src, t_new)
                    re[k], _ = resample(es[k], t_src, t_new)
            Tn = len(src_i)
            if Tn < a.H:
                continue
            ti = len(trajs)
            trajs.append(r["tid"])
            for t0 in range(0, Tn - a.H + 1, a.stride):
                if m == "chunk":
                    w = src_i[t0:t0 + a.H]
                    if bad[w.min():min(w.max() + 2, T)].any():
                        continue
                for k in out_hands:
                    q = rq[k][t0:t0 + a.H]
                    Q[k].append(q if order[k] is None else q[:, order[k]])
                    E[k].append(float(re[k][t0:t0 + a.H].mean()))
                feix.append(r["feix"]); source.append(r["src"]); traj.append(ti); t0s.append(t0); split.append(sp)
                stretch.append(s)
    hands = out_hands
    out = {f"q_{k}": np.asarray(Q[k], np.float32) for k in hands}
    out.update({f"err_{k}": np.asarray(E[k], np.float32) for k in hands})
    out.update(feix=np.asarray(feix, np.int16), source=np.asarray(source, np.int8), traj=np.asarray(traj, np.int32),
               t0=np.asarray(t0s, np.int32), split=np.asarray(split, np.int8), stretch=np.asarray(stretch, np.float16))
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
    meta = {"hands": hands, "finger_joint_names": names, "H": a.H, "stride": a.stride, "fps": fps, "fps_source": fps_in,
            "bench": a.bench or None, "retarget_hand": hmap, "stretch": [s_lo, s_hi],
            "order": {k: [int(j) for j in v] for k, v in order.items() if v is not None},
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
