"""Demo-retarget pairs for the pair-clustering metric (2026-11-19): per source hand, 400 task-demo frames (fingertips
>= 5 mm apart within the hand) retargeted to every other hand with DexPilot (cross_dex.ours.uniform_pairs.retarget,
300 it, 2 starts). Writes <out>/inputs.npz = <src>/inputs.npz with pair_<h> replaced by these tuples (+ pair_src,
pair_res_mm_<h>), so eval_space / zdump read it unchanged.  usage: make_demo_pairs.py <src eval dir> <out dir>"""
import sys, json, os, numpy as np, torch
sys.path.insert(0, "/workspace/codes/cross_dex/cross_dex/baselines/lad"); sys.path.insert(0, "/workspace/codes/cross_dex")
from dexverse_lad import eval_space as E
from dexverse_xlvla import hands as H
from cross_dex.ours import uniform_pairs as UP
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
I0 = dict(np.load(f"{src}/inputs.npz", allow_pickle=True)); I = E.flat(dict(I0))
hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = {h: H.CanonicalHandModel(E.model_name(h, side, E.bench_of(I))) for h in hands}
idx = {h: M[h].data_index(names[h]) for h in hands}
DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}
rng = np.random.default_rng(0); torch.manual_seed(0)
N, MIN_MM = 400, 5.0
def to_norm(h, q): return M[h].angles_to_normalized(torch.tensor(q[:, idx[h]])).clamp(-1, 1)
def to_data(h, qn):
    lo, hi = M[h]._lower.cpu().numpy(), M[h]._upper.cpu().numpy()
    ang = (qn + 1) * 0.5 * (hi - lo) + lo
    d = np.zeros((len(qn), len(names[h])), np.float32); d[:, idx[h]] = ang; return d
rows = {h: [] for h in hands}; res = {h: [] for h in hands}; srcs = []; stat = {}
with torch.no_grad():
    for si, s in enumerate(hands):
        qn = to_norm(s, I[f"teleop_{s}"]); T = M[s].forward(qn)                 # (n, k, 3)
        mind = torch.full((len(qn),), float("inf")); keep = []
        order = rng.permutation(len(qn))
        for i in order:                                                       # greedy: >= MIN_MM from kept ones
            if mind[i] < MIN_MM: continue
            keep.append(i); mind = torch.minimum(mind, (T - T[i]).norm(dim=-1).mean(-1) * 1e3)
            if len(keep) == N: break
        keep = np.array(keep); qs = qn[keep]
        rows[s].append(to_data(s, qs.numpy())); res[s].append(np.zeros(len(keep), np.float32)); srcs += [si] * len(keep)
        for t in hands:
            if t == s: continue
            with torch.enable_grad():
                q, e, _ = UP.retarget(M[s], M[t], qs.to(dev), 300, 2, 0.05, 1e-6, DP)
            rows[t].append(to_data(t, q.detach().cpu().numpy())); res[t].append(e.cpu().numpy())
            stat[f"{s}>{t}"] = round(float(e.median()), 1)
        print(f"[{s}] {len(keep)} frames kept, residual mm " + json.dumps({k: v for k, v in stat.items() if k.startswith(s + '>')}), flush=True)
for h in hands:
    I0[f"pair_{h}"] = np.concatenate(rows[h]); I0[f"pair_res_mm_{h}"] = np.concatenate(res[h])
I0["pair_src"] = np.array(srcs)
np.savez(f"{out}/inputs.npz", **I0)
json.dump({"src": src, "n_per_source": N, "min_mm": MIN_MM, "dexpilot": DP, "iters": 300, "starts": 2, "residual_median_mm": stat},
          open(f"{out}/demo_pairs.meta.json", "w"), indent=1)
print("[done]", out, {h: I0[f"pair_{h}"].shape for h in hands}, flush=True)
