"""Hand closing / opening durations of DexCanvas trajectories, per grasp type -> durations for synthesized transitions.

Aperture a(t) = mean distance of the 5 fingertips to the palm centre (wrist + 4 long-finger MCPs). Closing = the largest
drop of a(t): time from a crossing (a_max - 10% drop) to (a_min + 10% drop) (10-90 % time, robust to holds), a_max being
the maximum before the minimum. Opening = the same for the largest rise after the minimum. Small changes (< --min_drop)
are not counted as a closing/opening.

  /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.motion_timing --shards 0:894:9 --out timing.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from cross_dex.dataset.chunk_paired import DATA_ROOT  # noqa: E402

PALM = [0, 1, 4, 7, 10]          # DexCanvas order: palm(wrist), index/middle/ring/pinky MCP
TIPS = [16, 17, 18, 19, 20]


def aperture(j: np.ndarray) -> np.ndarray:
    c = j[:, PALM].mean(1, keepdims=True)
    return np.linalg.norm(j[:, TIPS] - c, axis=-1).mean(1)


def ten_ninety(t: np.ndarray, a: np.ndarray, hi: float, lo: float, i0: int, i1: int) -> float | None:
    """Time a(t) needs to go from hi - 10% to lo + 10% (or reverse) inside [i0, i1]."""
    d = hi - lo
    seg_t, seg_a = t[i0: i1 + 1], a[i0: i1 + 1]
    if seg_a[0] > seg_a[-1]:      # closing
        s = np.argmax(seg_a <= hi - 0.1 * d)
        e = np.argmax(seg_a <= lo + 0.1 * d)
    else:                          # opening
        s = np.argmax(seg_a >= lo + 0.1 * d)
        e = np.argmax(seg_a >= hi - 0.1 * d)
    return float(seg_t[e] - seg_t[s]) if e > s else None


def feix_of(label: str, table: dict[int, int]) -> int:
    k = int(label.split("-")[0]) if label[:1].isdigit() else -1
    return table.get(k, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA_ROOT, "dexcanvas", "keypoints"))
    ap.add_argument("--shards", default="0:894:9", help="start:stop:step")
    ap.add_argument("--min_drop", type=float, default=0.01, help="m, smaller aperture changes are no closing/opening")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # DexCanvas label number -> Feix number (FEIX_MAPPING.md; Prismatic n counts the thumb)
    table = {1: 9, 2: 8, 3: 7, 4: 6, 7: 20, 8: 22, 9: 1, 10: 18, 11: 28, 12: 19, 13: 31, 14: 4, 15: 17, 16: 23, 18: 16, 19: 25}
    s0, s1, st = (int(x) for x in a.shards.split(":"))
    files = sorted(glob.glob(os.path.join(a.data, "train-*.npz")))[s0:s1:st]
    rec = {"close": {}, "open": {}, "drop_m": {}}
    bad = []
    for f in files:
        try:
            with np.load(f) as zf:
                z = {k: zf[k] for k in ("offsets", "action_label", "t", "joints")}
        except Exception as e:  # noqa: BLE001  corrupt shard -> skip, report
            bad.append(f"{os.path.basename(f)}: {e!r}")
            continue
        off = z["offsets"]
        for i, lab in enumerate(z["action_label"].tolist()):
            g = feix_of(str(lab), table)
            if g < 0:
                continue
            t = z["t"][off[i]: off[i + 1]].astype(np.float64)
            ap_ = aperture(z["joints"][off[i]: off[i + 1]].astype(np.float64))
            im = int(np.argmin(ap_))
            ih = int(np.argmax(ap_[: im + 1]))
            drop = ap_[ih] - ap_[im]
            if drop >= a.min_drop:
                d = ten_ninety(t, ap_, ap_[ih], ap_[im], ih, im)
                if d:
                    rec["close"].setdefault(g, []).append(d)
                    rec["drop_m"].setdefault(g, []).append(float(drop))
            if im < len(ap_) - 1:
                io = im + int(np.argmax(ap_[im:]))
                rise = ap_[io] - ap_[im]
                if rise >= a.min_drop:
                    d = ten_ninety(t, ap_, ap_[io], ap_[im], im, io)
                    if d:
                        rec["open"].setdefault(g, []).append(d)
    summ = {}
    for kind in ("close", "open", "drop_m"):
        allv = np.concatenate([np.asarray(v) for v in rec[kind].values()]) if rec[kind] else np.zeros(0)
        summ[kind] = {"all": {"n": int(len(allv)), "p10_p50_p90": np.round(np.percentile(allv, [10, 50, 90]), 3).tolist()} if len(allv) else {}}
        for g, v in sorted(rec[kind].items()):
            summ[kind][g] = {"n": len(v), "p10_p50_p90": np.round(np.percentile(v, [10, 50, 90]), 3).tolist()}
    summ["args"] = vars(a)
    summ["bad_files"] = bad
    with open(a.out, "w") as f:
        json.dump(summ, f, indent=1)
    print("bad files:", bad)
    for kind in ("close", "open", "drop_m"):
        print(kind, "all", summ[kind]["all"])
        for g in sorted(k for k in summ[kind] if k != "all"):
            print(f"  feix {g:2d}", summ[kind][g])


if __name__ == "__main__":
    main()
