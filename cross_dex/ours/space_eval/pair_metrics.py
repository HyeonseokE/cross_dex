"""Pair clustering ("쌍 뭉침") metrics of one space from its latents on the demo-pair and bundle sets (2026-11-19).
Same formulas as the artifact tables (zspace analyze_demo / analyze_resbin / analyze_bundle), numpy only.

  python pair_metrics.py --demopair <dir with inputs.npz> --z_demopair <z_<name>.npz> \
                         --bundle <dir with inputs.npz>   --z_bundle <z_<name>.npz> [--json out.json]

Retrieval: for every ordered hand pair (A, B), A != B, each pair row's z on hand A ranks all rows' z on hand B by L2
distance; rank 1 = its own partner is the nearest. Numbers are means over the ordered pairs (both directions).
  r1, r10, mrank   demo-pair set, all rows (2,400 on DexVerse right: 400 demo frames per source hand x 6 hands)
  prec             demo-pair set, R@1 of the query rows whose retargeting residual is < 5 mm ("정밀도 R@1")
                   (rows whose source is A or B; residual = DexPilot residual of the retargeted partner)
  bund             bundle set: candidates = the 5 rows of the query's bundle (5 poses 1-5 mm apart, chance 20 %)
                   ("손 간 해상도 %")
"""
import argparse, json
import numpy as np


def cd(A, B):
    return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0))


def load(inputs, zfile):
    I = np.load(f"{inputs}/inputs.npz", allow_pickle=True); Z = np.load(zfile)
    H = [str(h) for h in I["hands"]]
    P = {h: Z[f"pair_{h}"].astype(np.float64) for h in H}
    res = {h: I[f"pair_res_mm_{h}"] for h in H}
    return H, P, I["pair_src"], res, (I["pair_bundle"] if "pair_bundle" in I.files else None)


def demopair(inputs, zfile):
    H, P, src, R, _ = load(inputs, zfile)
    r1, r10, mr, pr = [], [], [], []
    for ia, a in enumerate(H):
        for ib, b in enumerate(H):
            if a == b:
                continue
            Dm = cd(P[a], P[b]); rank = (Dm < np.diag(Dm)[:, None]).sum(1) + 1
            r1.append((rank <= 1).mean()); r10.append((rank <= 10).mean()); mr.append(np.median(rank))
            res = np.where(src == ia, R[b], np.where(src == ib, R[a], np.nan))
            m = (res >= 0) & (res < 5)
            if m.sum() >= 10:
                pr.append((rank[m] == 1).mean())
    return dict(r1=100 * float(np.mean(r1)), r10=100 * float(np.mean(r10)), mrank=float(np.mean(mr)),
                prec=100 * float(np.mean(pr)) if pr else float("nan"))


def bundle(inputs, zfile):
    H, P, src, R, B = load(inputs, zfile)
    same = B[:, None] == B[None]; acc = []
    for ia, a in enumerate(H):
        for ib, b in enumerate(H):
            if a == b:
                continue
            Dm = cd(P[a], P[b]); pos = np.diag(Dm)
            brank = ((Dm < pos[:, None]) & same).sum(1) + 1
            q = (src == ia) | (src == ib)
            acc.append((brank[q] == 1).mean())
    return dict(bund=100 * float(np.mean(acc)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demopair"); ap.add_argument("--z_demopair")
    ap.add_argument("--bundle"); ap.add_argument("--z_bundle")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    if a.demopair:
        out.update(demopair(a.demopair, a.z_demopair))
    if a.bundle:
        out.update(bundle(a.bundle, a.z_bundle))
    print(json.dumps({k: round(v, 2) for k, v in out.items()}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
