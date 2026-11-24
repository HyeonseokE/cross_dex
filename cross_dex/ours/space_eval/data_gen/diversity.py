import sys, json, glob, numpy as np, torch
sys.path.insert(0, "/workspace/codes/cross_dex/cross_dex/baselines/lad")
from dexverse_lad import eval_space as E
from dexverse_xlvla import hands as H
from dexverse_xlvla import eval_space_dexjoco as EJ
torch.set_grad_enabled(False)
for b in ("dv_right", "dj_right"):
    I = E.flat(E.load_inputs(f"/workspace/datasets/dexverse/lad/eval_space/{b}")[0])
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
    for h in hands:
        m = H.CanonicalHandModel(E.model_name(h, side, E.bench_of(I))); idx = m.data_index(names[h])
        q = I[f"teleop_{h}"]; T = m.forward(m.angles_to_normalized(torch.tensor(q[:, idx])).clamp(-1, 1))
        rng = np.random.default_rng(0); mind = torch.full((len(T),), float("inf")); kept = 0
        for i in rng.permutation(len(T)):
            if mind[i] < 5: continue
            kept += 1; mind = torch.minimum(mind, (T - T[i]).norm(dim=-1).mean(-1) * 1e3)
        jstd = float(np.degrees(q[:, idx].std(0)).mean())
        extra = ""
        if b == "dj_right":
            fs = sorted(glob.glob(f"{EJ.DEMO_ROOT}/*/{EJ.HD.FILE[h]}/scene_*/native.npz"))
            tasks = {f.split('/')[-4] for f in fs}; nfr = sum(int(np.load(f)['action'].shape[0]) for f in fs[:400])
            extra = f" | files {len(fs)} tasks {len(tasks)} frames(first400 files) {nfr}"
        print(f"{b} {h}: frames {len(q)}, distinct@5mm {kept}, mean joint std {jstd:.1f} deg{extra}", flush=True)
