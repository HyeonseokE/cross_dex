import sys, json, numpy as np, torch
sys.path.insert(0, "/workspace/codes/cross_dex/cross_dex/baselines/lad"); sys.path.insert(0, "/workspace/codes/cross_dex")
from dexverse_lad import eval_space as E
from dexverse_xlvla import hands as H
from cross_dex.ours import uniform_pairs as UP
I = E.flat(E.load_inputs(sys.argv[1])[0])
hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
M = {h: H.CanonicalHandModel(E.model_name(h, side, E.bench_of(I))) for h in hands}
print({h: M[h].hand_name for h in hands})
idx = {h: M[h].data_index(names[h]) for h in hands}
sub = np.arange(len(I[f"pair_{hands[0]}"]))
DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}
torch.set_grad_enabled(False)
q = {h: M[h].angles_to_normalized(torch.tensor(I[f"pair_{h}"][sub][:, idx[h]])).clamp(-1, 1) for h in hands}
kv = {h: UP.keyvecs(M[h], q[h]) for h in hands}
uq = {h: torch.empty(10, M[h].dof_count()).uniform_(-1, 1) for h in hands}
print("dir | eval pair residual mm p25/p50/p75 | geometric oracle R@1 % / median rank | random-pose-to-random residual p50")
allres, r1s, mrs = [], [], []
for s in hands:
    for t in hands:
        if s == t: continue
        k = min(M[s].tip_count(), M[t].tip_count())
        r_i = UP.open_lengths(M[t])[:k] / UP.open_lengths(M[s])[:k]
        sv, n_ff = UP.dexpilot_vectors(*kv[s], k)
        ref, w, _ = UP.dexpilot_reference(sv, n_ff, k, r_i, r_i.mean(), DP["project_dist"], DP["eta1"], DP["eta2"])
        tv, _ = UP.dexpilot_vectors(*kv[t], k)
        res = (tv - ref).norm(dim=-1).pow(2).mean(-1).sqrt() * 1000                      # same pose
        # oracle: unweighted RMS residual between source i's reference and every target j
        Dm = torch.cdist(ref.reshape(len(sub), -1), tv.reshape(len(sub), -1)) / np.sqrt(ref.shape[1]) * 1000
        pos = Dm.diag(); rank = (Dm < pos[:, None]).sum(1) + 1
        r1s.append(float((rank == 1).float().mean())); mrs.append(float(rank.float().median()))
        allres.append(res)
        p = lambda x, a: float(np.percentile(x.numpy(), a))
        print(f"{s}>{t} | {p(res,25):.1f}/{p(res,50):.1f}/{p(res,75):.1f} | {100*r1s[-1]:.1f} / {mrs[-1]:.0f}", flush=True)
a = torch.cat(allres)
print(f"ALL | eval pair residual p50 {float(a.median()):.1f} mm, mean {float(a.mean()):.1f} | oracle R@1 {100*np.mean(r1s):.1f}% median rank {np.mean(mrs):.1f}")
