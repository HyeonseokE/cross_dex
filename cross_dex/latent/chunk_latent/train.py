"""Train the cross-embodiment chunk latent (model.py) on a build_chunks.py dataset.

Stage 1 (--steps1): encoders only, contrastive. Stage 2 (--steps2): encoders + decoders, recon + cross recon +
w_contr * contrastive (LAD order: contrastive first, then decoders with encoder fine-tuning). Type-balanced sampling
(each Feix type equally likely) by default. Writes the experiment-rule files into --run_dir (config.yaml, meta.json,
metrics.csv, settings.md; stdout -> logs/ by the caller) and best/last checkpoints into --ckpt_dir.

  PYTHONPATH=/workspace/codes/cross_dex:/workspace/experiments /workspace/envs/dexverse/lerobot/bin/python \
      -m cross_dex.latent.chunk_latent.train --data <chunks dir> --run_dir <run dir> --ckpt_dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from cross_dex.latent.chunk_latent.model import ChunkLatent, losses, tau_at

KST = dt.timezone(dt.timedelta(hours=9))


def now():
    return dt.datetime.now(KST).isoformat(timespec="seconds")


def git_commit(path):
    try:
        c = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        d = subprocess.call(["git", "-c", "safe.directory=*", "-C", path, "diff", "--quiet"]) != 0
        return c, d
    except Exception:  # noqa: BLE001
        return None, None


@torch.no_grad()
def evaluate(model, X, idx, mean, std, hands, bs=4096):
    """Val metrics: per-hand recon / cross-recon MAE (deg, un-normalised), retrieval top-1, jerk ratio."""
    model.eval()
    acc = {}
    n = 0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x = {h: X[h][b] for h in hands}
        z = {h: model.encode(h, x[h]) for h in hands}
        for h in hands:
            for g in hands:
                y = model.decode(g, z[h]) * std[g] + mean[g]
                gt = x[g] * std[g] + mean[g]
                key = f"mae_deg/{h}->{g}"
                acc[key] = acc.get(key, 0.0) + float((y - gt).abs().mean()) * len(b) * 57.2958
                if h == g and y.shape[1] >= 4:
                    jy = (y[:, 3:] - 3 * y[:, 2:-1] + 3 * y[:, 1:-2] - y[:, :-3]).abs().mean()
                    jg = (gt[:, 3:] - 3 * gt[:, 2:-1] + 3 * gt[:, 1:-2] - gt[:, :-3]).abs().mean()
                    acc[f"jerk_ratio/{h}"] = acc.get(f"jerk_ratio/{h}", 0.0) + float(jy / jg.clamp_min(1e-9)) * len(b)
        n += len(b)
    out = {k: v / n for k, v in acc.items()}
    # retrieval: does hand a's latent find the paired chunk of hand b among 1024 val chunks
    b = idx[:1024]
    z = {h: torch.nn.functional.normalize(model.encode(h, X[h][b]).flatten(1), dim=-1) for h in hands}
    r = []
    for i, a in enumerate(hands):
        for c in hands[i + 1:]:
            r.append(float(((z[a] @ z[c].t()).argmax(1) == torch.arange(len(b), device=z[a].device)).float().mean()))
    out["retrieval_top1"] = float(np.mean(r))
    out["mae_deg/self"] = float(np.mean([out[f"mae_deg/{h}->{h}"] for h in hands]))
    out["mae_deg/cross"] = float(np.mean([out[f"mae_deg/{h}->{g}"] for h in hands for g in hands if g != h]))
    model.train()
    return out


def write_settings(run_dir, a, meta, n_tr, n_va, dofs, n_params, final, best_step):
    L = ["# settings", "", "## 태스크·데이터",
         f"- 태스크: 손 4종 교차 체현 청크 잠재 행동 공간 학습 (짝지은 리타게팅 청크)",
         f"- 데이터: `{a.data}` (build_chunks: H={meta['H']}, stride={meta['stride']}, {meta['fps']} Hz, 출처 {meta['sources']})",
         f"- train/val: {n_tr} / {n_va} 청크 (궤적 단위 분할), Feix 종류 {len(meta['feix_types'])}개, 종류 균형 샘플링 {a.balance}",
         "", "## 입력 (Input)", f"- 이미지 없음. 손별 손가락 목표 관절 청크 [H={meta['H']}, n]: " +
         ", ".join(f"{h} {n}" for h, n in dofs.items()), "- 정규화: 손별 관절마다 train 평균·표준편차 z-score",
         "", "## 출력 (Output)", f"- 잠재 [K={a.K}, d={a.d}] (시간 압축 {meta['H']}→{a.K}), 디코더 출력 = 각 손 청크 [H, n] (절대 관절 목표)",
         "", "## 모델", f"- {'VAE (토큰별 가우시안 사후분포, KL β=' + str(a.beta) + ', 정책 데이터·평가는 평균)' if a.vae else 'AE (결정적)'}; 손별 Transformer 인코더({a.layers}층, 폭 {a.width}) + strided conv {meta['H']}→{a.K} / 손별 Transformer 디코더, 파라미터 {n_params:,}",
         "", "## 학습 하이퍼파라미터",
         f"- AdamW lr {a.lr}, wd {a.wd}, batch {a.batch}, 1단계(대조) {a.steps1} step, 2단계(재구성+교차재구성+{a.w_contr}×대조, 교차 가중 {a.w_cross}, 속도 항 {a.w_vel}) {a.steps2} step",
         f"- InfoNCE 양방향, τ 0.25→0.16 지수 감소, seed {a.seed}, best = val 교차 재구성 MAE 최소 (step {best_step})",
         f"- 학습 코드 commit: {git_commit(os.path.dirname(os.path.abspath(__file__)))[0]}",
         "", "## 추론(평가) 하이퍼파라미터", "- 정책 평가 없음 (잠재 공간 자체 평가). val 전체:",
         *[f"  - {k}: {v:.4f}" for k, v in sorted(final.items())]]
    open(os.path.join(run_dir, "settings.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--steps1", type=int, default=10000)
    ap.add_argument("--steps2", type=int, default=40000)
    ap.add_argument("--w_contr", type=float, default=0.1)
    ap.add_argument("--w_cross", type=float, default=1.0)
    ap.add_argument("--teleop", default="", help="comma list of teleop_chunks.py npz mixed into stage 2 (self recon only, unpaired)")
    ap.add_argument("--w_teleop", type=float, default=1.0)
    ap.add_argument("--teleop_test", default="", help="teleop_chunks.py npz used only as a test set (val metric teleop_mae_deg)")
    ap.add_argument("--frame", type=int, default=0, help="1 = single-frame baseline: H=1, K=1 (first frame of each chunk)")
    ap.add_argument("--vae", type=int, default=0, help="1 = Gaussian posterior + KL (beta)")
    ap.add_argument("--beta", type=float, default=1e-5)
    ap.add_argument("--w_vel", type=float, default=0.0, help="velocity-matching term in recon / cross recon (smoothness)")
    ap.add_argument("--balance", type=int, default=1, help="1 = each Feix type equally likely")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    from seed_control import DEFAULT_SEED, seed_everything   # experiments/seed_control.py (EXPERIMENT_RULES 4)
    a.seed = seed_everything(DEFAULT_SEED if a.seed is None else a.seed)
    os.makedirs(a.run_dir, exist_ok=True)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    run_id = os.path.basename(a.run_dir.rstrip("/"))
    commit, dirty = git_commit(os.path.dirname(os.path.abspath(__file__)))
    meta_run = {"run_id": run_id, "project": os.path.basename(os.path.dirname(a.run_dir.rstrip("/"))),
                "server": os.environ.get("SERVER", socket.gethostname()), "agent": "claude-code",
                "git_repo": "HyeonseokE/cross_dex", "git_commit": commit, "git_dirty": dirty, "docker_image": None,
                "command": " ".join(sys.argv), "seed": a.seed, "start_time": now(), "end_time": None, "status": "running"}
    json.dump(meta_run, open(os.path.join(a.run_dir, "meta.json"), "w"), indent=1)
    open(os.path.join(a.run_dir, "config.yaml"), "w").write("".join(f"{k}: {v}\n" for k, v in vars(a).items()))
    mpath = os.path.join(a.run_dir, "metrics.csv")
    open(mpath, "w").write("step,split,metric,value\n")

    dmeta = json.load(open(os.path.join(a.data, "meta.json")))
    z = np.load(os.path.join(a.data, "chunks.npz"))
    hands = dmeta["hands"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = {h: torch.tensor(dmeta["norm"][h]["mean"], device=dev) for h in hands}
    std = {h: torch.tensor(dmeta["norm"][h]["std"], device=dev).clamp_min(1e-3) for h in hands}
    X = {h: (torch.from_numpy(z[f"q_{h}"]).to(dev) - mean[h]) / std[h] for h in hands}
    if a.frame:                                   # same data / losses, one frame in, one latent token out
        X = {h: x[:, :1] for h, x in X.items()}
        dmeta["H"], a.K = 1, 1
    split, feix = z["split"], z["feix"]
    tr_idx, va_idx = np.where(split == 0)[0], np.where(split == 1)[0]
    if a.balance:
        types, inv = np.unique(feix[tr_idx], return_inverse=True)
        w = 1.0 / np.bincount(inv)[inv]
        p = torch.tensor(w / w.sum(), device=dev, dtype=torch.float)
    def load_tel(paths):
        out = {h: [] for h in hands}
        for pth in [p for p in paths.split(",") if p]:
            zt = np.load(pth)
            for h in hands:
                if f"q_{h}" in zt.files:
                    out[h].append(zt[f"q_{h}"])
        return {h: ((torch.from_numpy(np.concatenate(v)).to(dev) - mean[h]) / std[h])[:, :dmeta["H"]] for h, v in out.items() if v}
    TEL = load_tel(a.teleop)
    TEST = load_tel(a.teleop_test)
    print("teleop train chunks", {h: len(v) for h, v in TEL.items()}, "teleop test chunks", {h: len(v) for h, v in TEST.items()}, flush=True)
    tr_t = torch.from_numpy(tr_idx).to(dev)
    va_t = torch.from_numpy(va_idx).to(dev)
    dofs = {h: X[h].shape[-1] for h in hands}
    model = ChunkLatent(dofs, dmeta["H"], a.K, a.d, a.width, a.layers, vae=bool(a.vae)).to(dev)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"train {len(tr_idx)} val {len(va_idx)} dofs {dofs} params {n_params}", flush=True)
    total = a.steps1 + a.steps2
    best, best_step, final = float("inf"), 0, {}
    opt = torch.optim.AdamW(model.enc.parameters(), lr=a.lr, weight_decay=a.wd)
    t0 = time.time()
    for step in range(1, total + 1):
        stage = 1 if step <= a.steps1 else 2
        if step == a.steps1 + 1:      # stage 2: decoders join, encoders fine-tuned
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps2)
        bi = tr_t[torch.multinomial(p, a.batch, replacement=True)] if a.balance else \
            tr_t[torch.randint(len(tr_t), (a.batch,), device=dev)]
        L = losses(model, {h: X[h][bi] for h in hands}, tau_at(step, total), stage, a.w_contr, a.w_cross, a.w_vel, a.beta)
        if stage == 2 and TEL:
            from cross_dex.latent.chunk_latent.model import _rec
            tl = []
            for h, xt in TEL.items():
                xb = xt[torch.randint(len(xt), (a.batch // 4,), device=dev)]
                mu, lv = model.posterior(h, xb)
                zb = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if model.vae else mu
                tl.append(_rec(model.decode(h, zb), xb, a.w_vel))
            L["teleop"] = sum(tl) / len(tl)
            L["loss"] = L["loss"] + a.w_teleop * L["teleop"]
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if stage == 2:
            sched.step()
        if step % 200 == 0:
            with open(mpath, "a") as f:
                for k, v in L.items():
                    f.write(f"{step},train,{k},{float(v):.6f}\n")
        if step % a.eval_every == 0 or step == total:
            ev = evaluate(model, X, va_t, mean, std, hands)
            if TEST:
                with torch.no_grad():
                    model.eval()
                    for h, xt in TEST.items():
                        e = []
                        for i in range(0, len(xt), 4096):
                            xb = xt[i:i + 4096]
                            e.append(((model.decode(h, model.encode(h, xb)) - xb).abs() * std[h]).mean(dim=(1, 2)))
                        ev[f"teleop_mae_deg/{h}"] = float(torch.cat(e).mean()) * 57.2958
                    ev["teleop_mae_deg/mean"] = float(np.mean([ev[f"teleop_mae_deg/{h}"] for h in TEST]))
                    model.train()
            with open(mpath, "a") as f:
                for k, v in ev.items():
                    f.write(f"{step},val,{k},{v:.6f}\n")
            print(f"[{step}/{total}] stage {stage} {time.time() - t0:.0f}s " + " ".join(f"{k}={float(v):.4f}" for k, v in L.items())
                  + f" | val self {ev['mae_deg/self']:.2f}deg cross {ev['mae_deg/cross']:.2f}deg top1 {ev['retrieval_top1']:.3f}"
                  + (f" teleop {ev['teleop_mae_deg/mean']:.2f}deg " + str({h: round(ev[f'teleop_mae_deg/{h}'], 1) for h in TEST}) if TEST else ""), flush=True)
            ck = {"model": model.state_dict(), "args": vars(a), "dofs": dofs, "hands": hands, "H": dmeta["H"],
                  "norm": dmeta["norm"], "finger_joint_names": dmeta["finger_joint_names"], "data": a.data, "step": step, "val": ev}
            torch.save(ck, os.path.join(a.ckpt_dir, "last.pt"))
            if stage == 2 and ev["mae_deg/cross"] < best:
                best, best_step, final = ev["mae_deg/cross"], step, ev
                torch.save(ck, os.path.join(a.ckpt_dir, "best.pt"))
    write_settings(a.run_dir, a, dmeta, len(tr_idx), len(va_idx), dofs, n_params, final, best_step)
    meta_run.update(end_time=now(), status="done", best_step=best_step, best_val=final)
    json.dump(meta_run, open(os.path.join(a.run_dir, "meta.json"), "w"), indent=1)
    print("done", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
