"""Hand-agnostic task residual in fingertip space for an unseen hand (RSS2027, 2026-11-23; dexlatent venv).

Decomposition of the finger action for hand h:  q = IK_h( tips(D_h(z) + a_h(z)) + s_h * g(feature) )
  a_h  task-agnostic per-hand decoder correction (decoder_fix on uniform poses; needs only the hand model)  -> --agnostic
  g    hand-agnostic TASK residual: wrist->tip vectors of thumb..ring (k=4, the tips every hand has) normalized by the
       hand's open-hand finger length s_h; g maps the base pose's normalized vectors to the correction of those vectors
       (target: normalized vectors of the demo q* minus those of the base pose). Trained on the OTHER hands' task demos.
Unseen target t (default shadow): no data of t is used except its hand model. Evaluated on t's real GraspKettle demos:
  before   D_t(E_t(q*))                     agnostic  + a_t                 tipres  + a_t + g (IK on k tips)
  oracle   IK to t's true k tip vectors (limit of the IK step)
Fingertip error vs q* in mm (all tips, canonical frame), as residual_fix / decoder_fix.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m \
     cross_dex.ours.crosshand_dp.tipres --latent_ckpt <space model.pt> --agnostic <decoder_fix .pt> --json <m.json>
"""
import argparse
import json

import numpy as np
import torch

from cross_dex.ours.diag.decoder_fix import demo_q, encdec, mlp, tip_mm
from cross_dex.ours.uniform_pairs import keyvecs, open_lengths

TASK = "Dexverse-GraspKettle-v0"
HANDS = ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]
K = 4


def feat(M, q):
    _, w2t = keyvecs(M, q)
    return w2t[:, :K].reshape(len(q), -1) / open_lengths(M)[:K].mean()


def base(tr, A, h, q):
    z, s = encdec(tr, h, q)
    with torch.no_grad():
        return z, s, (s + A[h](z)).clamp(-1, 1)


def ik(M, q0, target, iters=300, reg=1e-3):
    u = torch.atanh(q0.clamp(-0.999, 0.999)).clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.02)
    for _ in range(iters):
        opt.zero_grad()
        q = torch.tanh(u)
        loss = ((feat(M, q) - target) ** 2).sum(-1).mean() + reg * ((q - q0) ** 2).sum(-1).mean()
        loss.backward(); opt.step()
    return torch.tanh(u).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--agnostic", required=True)
    ap.add_argument("--target", default="shadow")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    F = torch.load(a.agnostic, map_location="cpu", weights_only=False)
    assert F["latent_ckpt"] == a.latent_ckpt
    A = {}
    for h in HANDS:
        A[h] = mlp(F["dof"][h]); A[h].load_state_dict(F["state_dicts"][h]); A[h].eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    t = a.target; srcs = [h for h in HANDS if h != t]
    X, Y, per_src = [], [], {}
    for h in srcs:
        M = tr.hand_models[h]; q = torch.cat(list(demo_q(tr, a.task, h)[1].values()))
        _, _, qb = base(tr, A, h, q)
        with torch.no_grad():
            fb, fq = feat(M, qb), feat(M, q)
        X.append(fb); Y.append(fq - fb); per_src[h] = float((fq - fb).norm(dim=-1).mean())
    X, Y = torch.cat(X), torch.cat(Y)
    net = torch.nn.Sequential(torch.nn.Linear(3 * K, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(),
                              torch.nn.Linear(512, 3 * K)).to(dev)
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), 1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    Xd, Yd = X.to(dev), Y.to(dev)
    for it in range(a.steps):
        b = torch.randint(0, len(Xd), (4096,), device=dev)
        opt.zero_grad(); loss = ((net(Xd[b]) - Yd[b]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
    net = net.cpu().eval()
    with torch.no_grad():
        fit_r2 = float(1 - ((net(X) - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    Mt = tr.hand_models[t]; q = torch.cat(list(demo_q(tr, a.task, t)[1].values()))
    _, s, qb = base(tr, A, t, q)
    with torch.no_grad():
        fb, fq = feat(Mt, qb), feat(Mt, q)
        pred = net(fb)
        r2_t = float(1 - ((pred - (fq - fb)) ** 2).sum() / (((fq - fb) - (fq - fb).mean(0)) ** 2).sum())
    q_tip = torch.cat([ik(Mt, qb[i:i + 4000], fb[i:i + 4000] + pred[i:i + 4000]) for i in range(0, len(q), 4000)])
    q_orc = torch.cat([ik(Mt, qb[i:i + 4000], fq[i:i + 4000]) for i in range(0, len(q), 4000)])
    q_ik0 = torch.cat([ik(Mt, qb[i:i + 4000], fb[i:i + 4000]) for i in range(0, len(q), 4000)])   # IK side effect only
    res = {"target": t, "sources": srcs, "frames": len(q), "train_frames": len(X), "train_fit_r2": fit_r2,
           "target_residual_r2": r2_t, "src_task_residual_norm": per_src,
           "target_task_residual_norm": float((fq - fb).norm(dim=-1).mean()),
           "tip_mm_before": float(tip_mm(Mt, q, s).mean()), "tip_mm_agnostic": float(tip_mm(Mt, q, qb).mean()),
           "tip_mm_agnostic_ik_noop": float(tip_mm(Mt, q, q_ik0).mean()),
           "tip_mm_agnostic_plus_tipres": float(tip_mm(Mt, q, q_tip).mean()), "tip_mm_oracle_ik": float(tip_mm(Mt, q, q_orc).mean())}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
