"""Resolution set (2-b, 2026-11-19): per source hand, anchors = task-demo frames >= 10 mm apart (fingertips); each anchor
+ 4 neighbours of the same hand's demo frames 1-5 mm away = one bundle of 5 close poses. Every pose is retargeted to every
other hand with DexPilot's objective: the anchor from 0 (2 starts, 300 it, as uniform_pairs.retarget), its neighbours
warm-started at the anchor's solution (300 it) so that close source poses get consistently solved partners. Writes
<out>/inputs.npz = <src>/inputs.npz with pair_<h> replaced (+ pair_src, pair_bundle, pair_res_mm_<h>).
usage: make_bundle_pairs.py <src eval dir> <out dir> [anchors per source]"""
import sys, json, os, numpy as np, torch
sys.path.insert(0, "/workspace/codes/cross_dex/cross_dex/baselines/lad"); sys.path.insert(0, "/workspace/codes/cross_dex")
from dexverse_lad import eval_space as E
from dexverse_xlvla import hands as H
from cross_dex.ours import uniform_pairs as UP
src, out = sys.argv[1], sys.argv[2]; NA = int(sys.argv[3]) if len(sys.argv) > 3 else 80; NB = 4
os.makedirs(out, exist_ok=True)
I0 = dict(np.load(f"{src}/inputs.npz", allow_pickle=True)); I = E.flat(dict(I0))
hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = {h: H.CanonicalHandModel(E.model_name(h, side, E.bench_of(I))) for h in hands}
idx = {h: M[h].data_index(names[h]) for h in hands}
DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}
rng = np.random.default_rng(0); torch.manual_seed(0)
def to_norm(h, q): return M[h].angles_to_normalized(torch.tensor(q[:, idx[h]])).clamp(-1, 1)
def to_data(h, qn):
    lo, hi = M[h]._lower.cpu().numpy(), M[h]._upper.cpu().numpy()
    d = np.zeros((len(qn), len(names[h])), np.float32); d[:, idx[h]] = (qn + 1) * 0.5 * (hi - lo) + lo; return d

def solve(s, t, qs, u0, iters):
    """DexPilot cost of uniform_pairs.retarget, Adam on tanh-parametrised joints from u0."""
    ts, ws = UP.keyvecs(M[s], qs); k = min(ts.shape[1], M[t].tip_count())
    r_i = (UP.open_lengths(M[t])[:k] / UP.open_lengths(M[s])[:k]).to(qs)
    sv, n_ff = UP.dexpilot_vectors(ts, ws, k)
    ref, w, _ = UP.dexpilot_reference(sv, n_ff, k, r_i, r_i.mean(), DP["project_dist"], DP["eta1"], DP["eta2"])
    hub = torch.nn.SmoothL1Loss(beta=DP["huber_delta"], reduction="none"); nv = ref.shape[1]
    def cost(q):
        tt, wt = UP.keyvecs(M[t], q); tv, _ = UP.dexpilot_vectors(tt, wt, k); d = (tv - ref).norm(dim=-1)
        return (hub(d, torch.zeros_like(d)) * w).sum(-1) / nv, d
    u = u0.clone().requires_grad_(True); opt = torch.optim.Adam([u], lr=0.05)
    with torch.enable_grad():
        for _ in range(iters):
            opt.zero_grad(); q = torch.tanh(u); c, _ = cost(q); (c.sum() + 1e-6 * q.pow(2).sum()).backward(); opt.step()
    with torch.no_grad():
        q = torch.tanh(u); _, d = cost(q)
    return q.detach(), d.pow(2).mean(-1).sqrt() * 1000

rows = {h: [] for h in hands}; res = {h: [] for h in hands}; srcs, bund = [], []; stat = {}; nb = 0
with torch.no_grad():
    for si, s in enumerate(hands):
        qn = to_norm(s, I[f"teleop_{s}"]); T = M[s].forward(qn)
        mind = torch.full((len(qn),), float("inf")); anchors, members = [], []
        for i in rng.permutation(len(qn)):
            if mind[i] < 10: continue
            dist = (T - T[i]).norm(dim=-1).mean(-1) * 1e3
            cand = torch.nonzero((dist >= 1) & (dist <= 5)).flatten().numpy()
            if len(cand) < NB: continue
            # neighbours also >= 1 mm from each other: greedy over a shuffled candidate list
            pick = []
            for c in rng.permutation(cand):
                if all(float((T[c] - T[p]).norm(dim=-1).mean() * 1e3) >= 1 for p in pick): pick.append(int(c))
                if len(pick) == NB: break
            if len(pick) < NB: continue
            anchors.append(int(i)); members.append(pick); mind = torch.minimum(mind, dist)
            if len(anchors) == NA: break
        A = np.array(anchors); Nb = np.array(members)                       # (na,), (na, NB)
        allq = torch.cat([qn[A]] + [qn[Nb[:, j]] for j in range(NB)])     # anchors first, then neighbour j of every anchor
        na = len(A)
        rows[s].append(to_data(s, allq.numpy())); res[s].append(np.zeros(len(allq), np.float32))
        srcs += [si] * len(allq); bund += list(nb + np.tile(np.arange(na), NB + 1)); nb += na
        for t in hands:
            if t == s: continue
            qa = qn[A].to(dev); best_q, best_e = None, None
            for st in range(2):                                            # anchors: 2 starts from 0 / random
                u0 = torch.zeros(na, M[t].dof_count(), device=dev) if st == 0 else torch.empty(na, M[t].dof_count(), device=dev).uniform_(-1.5, 1.5)
                q, e = solve(s, t, qa, u0, 300)
                if best_q is None: best_q, best_e = q, e
                else: m = e < best_e; best_q[m], best_e[m] = q[m], e[m]
            u_a = torch.atanh(best_q.clamp(-0.999, 0.999))
            qs_, es_ = [best_q], [best_e]
            for j in range(NB):                                            # neighbours warm-started at the anchor's partner
                q, e = solve(s, t, qn[Nb[:, j]].to(dev), u_a, 300); qs_.append(q); es_.append(e)
            rows[t].append(to_data(t, torch.cat(qs_).cpu().numpy())); res[t].append(torch.cat(es_).cpu().numpy())
            stat[f"{s}>{t}"] = round(float(torch.cat(es_).median()), 1)
        print(f"[{s}] {na} bundles x {NB + 1}, residual mm " + json.dumps({k: v for k, v in stat.items() if k.startswith(s + '>')}), flush=True)
for h in hands:
    I0[f"pair_{h}"] = np.concatenate(rows[h]); I0[f"pair_res_mm_{h}"] = np.concatenate(res[h])
I0["pair_src"] = np.array(srcs); I0["pair_bundle"] = np.array(bund)
np.savez(f"{out}/inputs.npz", **I0)
json.dump({"src": src, "anchors_per_source": NA, "neighbours": NB, "anchor_spacing_mm": 10, "neighbour_mm": [1, 5],
           "dexpilot": DP, "residual_median_mm": stat}, open(f"{out}/bundle_pairs.meta.json", "w"), indent=1)
print("[done]", out, {h: I0[f"pair_{h}"].shape for h in hands}, flush=True)
