"""Self-residual module on task data (RSS2027 detail restoration, 2026-11-23; dexlatent venv).

The shared space (E_h, D_h) and the multi-hand DP stay fixed. For each hand h a small MLP f_h(z) learns the self residual
of the task demos: r_h = q* - D_h(E_h(q*)), q* = h's recorded GraspKettle finger actions (teleop pickle, normalized joints).
At evaluation the policy's z is decoded as D_h(z) + f_h(z) (server.json "residual_fix", crosshand_dp/server.py).
Episodes --holdout (default 45-49) are left out of training and reported separately; the policy itself was trained on
all 50, so the module is fitted on the same demos the DP saw.
--train_tasks T1,T2,... fits f_h on those tasks' demos instead (task-holdout test, 2026-11-23: e.g. the other three
tasks, then evaluated on --task); when --task is not among them, all of --task's episodes count as held out.
Output file has the layout of cross_dex.ours.diag.decoder_fix (so its `build` and diag/pkl_replay.sh `fixed` also work).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m \
     cross_dex.ours.crosshand_dp.residual_fix --latent_ckpt <space model.pt> --out <fix.pt> [--json <m.json>]
"""
import argparse
import json

import numpy as np
import torch

from cross_dex.ours.diag.decoder_fix import demo_q, encdec, mlp, tip_mm

TASK = "Dexverse-GraspKettle-v0"
HANDS = "shadow,inspire,sharpa,wuji,allegro,leap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--hands", default=HANDS)
    ap.add_argument("--task", default=TASK, help="evaluation task (and the training task unless --train_tasks)")
    ap.add_argument("--train_tasks", default="", help="comma list; default = --task")
    ap.add_argument("--holdout", default="45-49")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load
    from dexverse_xlvla.replay import episode_list
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    hands = a.hands.split(","); hold = set(episode_list(a.holdout))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    out, nets = {}, {}
    for h in hands:
        M = tr.hand_models[h]; n = M.dof_count()
        tt = a.train_tasks.split(",") if a.train_tasks else [a.task]
        _, dq, _ = demo_q(tr, a.task, h)
        if a.task in tt:
            qtr = [v for k, v in dq.items() if k not in hold]; qte = torch.cat([v for k, v in dq.items() if k in hold])
        else:
            qtr = []; qte = torch.cat(list(dq.values()))
        for t in tt:
            if t != a.task:
                qtr += list(demo_q(tr, t, h)[1].values())
        qtr = torch.cat(qtr)
        ztr, str_ = encdec(tr, h, qtr); zte, ste = encdec(tr, h, qte)
        net = mlp(n).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        Z, R = ztr.to(dev), (qtr - str_).to(dev)
        for it in range(a.steps):
            b = torch.randint(0, len(Z), (4096,), device=dev)
            opt.zero_grad(); loss = ((net(Z[b]) - R[b]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
            if it % 5000 == 0:
                print(h, it, f"{loss.item():.2e}", flush=True)
        net = net.cpu().eval(); nets[h] = net
        with torch.no_grad():
            ftr = (str_ + net(ztr)).clamp(-1, 1); fte = (ste + net(zte)).clamp(-1, 1)
        out[h] = {"train_frames": len(qtr), "holdout_frames": len(qte),
                  "train_tip_mm_before": float(tip_mm(M, qtr, str_).mean()), "train_tip_mm_after": float(tip_mm(M, qtr, ftr).mean()),
                  "holdout_tip_mm_before": float(tip_mm(M, qte, ste).mean()), "holdout_tip_mm_after": float(tip_mm(M, qte, fte).mean())}
        print(h, json.dumps({k: round(v, 3) for k, v in out[h].items()}), flush=True)
    torch.save({"latent_ckpt": a.latent_ckpt, "pairs": None, "task": a.task, "train_tasks": a.train_tasks or a.task, "holdout": a.holdout, "hands": hands,
                "steps": a.steps, "kind": "self_residual_task",
                "state_dicts": {h: nets[h].state_dict() for h in hands},
                "dof": {h: tr.hand_models[h].dof_count() for h in hands}}, a.out)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
