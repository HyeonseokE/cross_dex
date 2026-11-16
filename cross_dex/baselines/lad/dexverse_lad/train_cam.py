"""LAD contrastive action model, trained as in the paper (Sec. III-B, Table II "mimic + Franka") with the authors' model
classes (robot_clip.model.RobotCLIP, robot_clip.objectives.info_nce) unchanged.

Paper procedure (the public scripts differ: two_step_train.py freezes the encoders in step 2 = Table III "no FT",
joint_train.py has no encoder pre-training and ignores the temperature schedule):
  stage 1: encoders only, pairwise InfoNCE (Eq. 4), temperature annealed exponentially 0.25 -> 0.16
  stage 2: decoders with reconstruction (Eq. 5) + encoders fine-tuned with a lower learning rate,
           L_total = L_recon + lambda * L_contrastive (Eq. 6), lambda = 0.1
Table II: batch 16384, AdamW, weight decay 1e-3, latent 16, MLP hidden 32,128,128,32, epochs 5000 (enc) / 10000
(dec + finetune), "Learning Rate 0.00001", "Finetuning Learning Rate 0.0001".
Interpretation (recorded in settings): stage 1 lr 1e-5; stage 2 decoders 1e-4 and encoders 1e-5 (lower, as the text
says). Not stated in the paper, taken from the public code: dropout 0.1, per-dimension z-score normalisation, random 80/20
frame split, decoder hidden dims = encoder hidden dims.
Early stopping (2026-11-15, user: "large epoch cap, stop when it no longer improves"; --patience > 0):
  --enc_epochs / --dec_epochs are upper bounds. Evaluation every --eval_every epochs on the held-out 20 % rows.
  stage 1: the temperature goes t0 -> t1 over --tau_epochs epochs, then stays at t1; from then on the stage stops when
           val_contrastive has not improved by --min_delta (relative) for --patience consecutive evaluations.
  stage 2: the same rule on val_SR_mean + val_CR_mean (self + cross reconstruction MSE).
  At a stop the weights of the best evaluation are restored. If a stage ends at its cap instead, it is logged
  ("reached the cap"), lad_cam_stage1_last.pt / lad_cam_last.pt keep the last-epoch weights and --resume continues the
  stage with a larger cap (optimizer and patience counters start fresh).
  Settings of the all-data runs: --enc_epochs 300 --dec_epochs 600 --tau_epochs 61 --eval_every 3 --patience 5 --min_delta 1e-3.
Validation: self-/cross-reconstruction (Eq. 7) for every hand pair, denormalised MSE in joint units.

  python -m dexverse_lad.train_cam --pairs <pairs.npy> --out <dir> [--latent 16 --enc_epochs 5000 --dec_epochs 10000]
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from robot_clip.model import RobotCLIP
from robot_clip.objectives import info_nce


class RunRecord:
    """Experiment record in --run_dir (EXPERIMENT_RULES.md): config.yaml, meta.json, metrics.csv. No-op without --run_dir."""

    def __init__(self, a, hands, dims, n, ntr):
        self.d = a.run_dir
        if not self.d:
            return
        import subprocess, sys
        os.makedirs(self.d, exist_ok=True)
        kst = lambda: time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.gmtime(time.time() + 9 * 3600))
        self.kst = kst
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        git = lambda *c: subprocess.run(["git", "-C", repo, *c], capture_output=True, text=True).stdout.strip()
        cfg = dict(vars(a), hands=hands, dims=dims, rows=n, train_rows=ntr, val_rows=n - ntr,
                   batches_per_epoch=math.ceil(ntr / a.batch))
        with open(f"{self.d}/config.yaml", "w") as f:
            f.write("".join(f"{k}: {json.dumps(v)}\n" for k, v in cfg.items()))
        self.meta = {"run_id": os.path.basename(self.d.rstrip("/")), "project": os.path.basename(os.path.dirname(self.d.rstrip("/"))),
                     "server": os.environ.get("SERVER"), "agent": "claude-code", "git_repo": git("remote", "get-url", "origin") or None,
                     "git_commit": git("rev-parse", "HEAD") or None, "git_dirty": bool(git("status", "--porcelain")),
                     "docker_image": None, "command": " ".join(sys.argv), "seed": a.seed, "start_time": kst(), "end_time": None,
                     "status": "running"}
        self._dump()
        self.csv = open(f"{self.d}/metrics.csv", "w"); self.csv.write("step,split,metric,value\n")
        self.bpe = cfg["batches_per_epoch"]; self.off = 0; self.last = 0      # last: stage 2 may be the first stage logged (--stage1_ckpt)

    def _dump(self):
        json.dump(self.meta, open(f"{self.d}/meta.json", "w"), indent=1)

    def metric(self, r):
        if not self.d:
            return
        if r["stage"] == 2 and not self.off:
            self.off = self.last + self.bpe
        step = (r["epoch"] + 1) * self.bpe + (self.off if r["stage"] == 2 else 0); self.last = step
        for k, v in r.items():
            if k not in ("stage", "epoch", "sec"):
                self.csv.write(f"{step},{'val' if k.startswith('val') else 'train'},stage{r['stage']}_{k},{v:.6g}\n")
        self.csv.flush()

    def note(self, **kw):
        if self.d:
            self.meta.setdefault("result", {}).update(kw); self._dump()

    def finish(self, **kw):
        if self.d:
            self.meta.setdefault("result", {}).update(kw); self.meta.update(end_time=self.kst(), status="done"); self._dump()


class Hooks:
    """Extension points for variants built on this reproduction (``cross_dex.ours.*``). The defaults are the LAD setting
    reproduced here, so ``python -m dexverse_lad.train_cam`` itself trains only LAD."""

    def add_args(self, ap):
        """Add the variant's own command-line options."""

    def setup(self, a, raw, hands, norm):
        """Called after the pairs are loaded; may return other normalisation stats {hand: {mean, std}}."""
        return norm

    def stage2_extra(self, a, model, hands, norm, dev):
        """Fine-tuned stage 2 only: None, or a function () -> (extra loss tensor, {metric: float}) added every batch."""
        return None


def make_model(dims, a):
    cfg = OmegaConf.create({"model": {
        "embedding_dim": a.latent, "encoder_dropout": a.dropout, "decoder_dropout": a.dropout,
        "modalities": {"modalities": {h: {"input_dim": d, "encoder_hidden_dims": a.hidden, "decoder_hidden_dims": a.hidden}
                                      for h, d in dims.items()}}}})
    return RobotCLIP(cfg), cfg


def temperature(e, n, t0, t1):
    return t0 * (t1 / t0) ** (e / max(n - 1, 1))


def pair_contrastive(z, tau):
    """Same value as RobotCLIP.contrastive_loss (InfoNCE i->j for every pair i<j, divided by M(M-1)), but the gradient
    w.r.t. each embedding is accumulated one pair at a time, so only one (B, B) logit matrix is alive at once
    (6 hands = 15 pairs; all at once at B=16384 does not fit a 32 GB GPU). Returns (loss value, {hand: dL/dz})."""
    hs = list(z); M = len(hs)
    zd = {h: v.detach().requires_grad_(torch.is_grad_enabled()) for h, v in z.items()}
    tot = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            l = info_nce(query=zd[hs[i]], positive_key=zd[hs[j]], negative_keys=None, temperature=tau,
                         reduction="mean") / (M * (M - 1))
            if torch.is_grad_enabled():
                l.backward()
            tot += float(l)
    return tot, ({h: v.grad for h, v in zd.items()} if torch.is_grad_enabled() else None)


@torch.no_grad()
def cross_recon(model, x, std):
    """Eq. 7 on held-out rows, denormalised (joint units^2): {src_to_dst: mse}."""
    model.eval()
    z = model.encode(x)
    out = {}
    for s in x:
        for t in x:
            d = (model.decoders[t](z[s]) - x[t]) * std[t]
            out[f"{s}_to_{t}"] = float((d ** 2).mean())
    model.train()
    return out


def main(argv=None, hooks=None):
    hooks = hooks or Hooks()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--hidden", default="32,128,128,32")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="stage 2 decoder lr (Table II 'Finetuning Learning Rate')")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--t0", type=float, default=0.25)
    ap.add_argument("--t1", type=float, default=0.16)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--enc_epochs", type=int, default=5000)
    ap.add_argument("--dec_epochs", type=int, default=10000)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th paired row (neighbouring frames are near-duplicates)")
    ap.add_argument("--left_override", default="",
                    help="with --left: hand=file.npy[,hand=file.npy] left-hand joint arrays (same rows, same joint order) that "
                         "replace the joint-wise conversion for those hands (DexJoCo-X left LEAP: leap_left_retarget.py)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping (2026-11-15): stop a stage when its validation loss has not improved by --min_delta "
                         "(relative) for this many evaluations and restore the best weights; 0 = run all epochs. With it, "
                         "--enc_epochs / --dec_epochs are upper bounds")
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--tau_epochs", type=int, default=0,
                    help="epochs of the temperature annealing t0 -> t1 (default: --enc_epochs); stage 1 early stopping starts "
                         "after it, at the constant temperature t1")
    ap.add_argument("--resume", default="",
                    help="lad_cam_last.pt of an interrupted / capped run: continue in its stage from the next epoch (same "
                         "data and seed -> same split; the optimizer and the early-stopping counters start fresh)")
    ap.add_argument("--stage1_ckpt", default="",
                    help="skip stage 1: load the encoders of a finished stage 1 (lad_cam_stage1_last.pt) and run stage 2 only "
                         "(used to train the two stage-2 variants from the same stage-1 encoders)")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="stage 2 trains the decoders only, encoders frozen (paper Table III 'no FT'; the public "
                         "two_step_train.py). Default: encoders fine-tuned at --lr with lam * contrastive (paper Sec. III-B)")
    ap.add_argument("--run_dir", default="", help="experiment run folder: config.yaml, meta.json, metrics.csv are written there")
    ap.add_argument("--left", action="store_true",
                    help="convert the (right-hand) pairs to the left-hand joint convention at load (build_pairs_bench.left_map, "
                         "needs <pairs>.meta.json); same data as build_pairs_bench's left file without storing it")
    hooks.add_args(ap)
    a = ap.parse_args(argv)
    a.hidden = [int(v) for v in a.hidden.split(",")]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    raw = np.load(a.pairs, allow_pickle=True).item()
    raw = {h: v[::a.stride] for h, v in raw.items()}
    if a.left:
        from dexverse_lad.build_pairs_bench import left_map
        pm = json.load(open(a.pairs[:-4] + ".meta.json"))
        for h in list(raw):
            idx, sgn = left_map(pm["retarget_hand"][h], pm["joint_names"][h], pm["bench"])
            raw[h] = np.ascontiguousarray(raw[h][:, idx] * sgn)
        for kv in [x for x in a.left_override.split(",") if x]:
            h, f = kv.split("=", 1)
            ov = np.load(f)[::a.stride]
            assert ov.shape == raw[h].shape, f"--left_override {h}: {ov.shape} vs {raw[h].shape}"
            raw[h] = np.ascontiguousarray(ov.astype(np.float32))
    hands = list(raw)
    n = len(raw[hands[0]])
    norm = {h: {"mean": raw[h].mean(0), "std": raw[h].std(0) + 1e-8} for h in hands}
    norm = hooks.setup(a, raw, hands, norm)
    perm = np.random.permutation(n); ntr = int(a.split * n)
    X = {h: torch.tensor((raw[h] - norm[h]["mean"]) / norm[h]["std"], dtype=torch.float32, device=dev) for h in hands}
    tr, va = torch.tensor(perm[:ntr], device=dev), torch.tensor(perm[ntr:], device=dev)
    Xva = {h: v[va] for h, v in X.items()}
    Xva_c = {h: v[:a.batch] for h, v in Xva.items()}  # val contrastive on one batch-sized held-out subset ((N_val)^2 logits do not fit)
    std = {h: torch.tensor(norm[h]["std"], dtype=torch.float32, device=dev) for h in hands}

    model, cfg = make_model({h: raw[h].shape[1] for h in hands}, a)
    model.to(dev)
    log = open(f"{a.out}/train_log.jsonl", "a")
    print(f"[lad] {n} paired rows ({ntr} train), hands {hands}, dims {[raw[h].shape[1] for h in hands]}, device {dev}", flush=True)

    def save(name, **extra):  # also written at every evaluation (lad_cam_last.pt): a long run keeps its latest weights
        torch.save({"state_dict": model.state_dict(), "config": OmegaConf.to_container(cfg), "args": vars(a), "hands": hands,
                    "normalization": {h: {k: v.tolist() for k, v in d.items()} for h, d in norm.items()}, **extra},
                   f"{a.out}/{name}.tmp")
        os.replace(f"{a.out}/{name}.tmp", f"{a.out}/{name}")

    def batches():
        p = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(p), a.batch):
            yield {h: v[p[i:i + a.batch]] for h, v in X.items()}

    run = RunRecord(a, hands, {h: int(raw[h].shape[1]) for h in hands}, n, ntr)
    tau_n = a.tau_epochs or a.enc_epochs

    class Stopper:  # best validation value, patience counter, best weights (CPU copy)
        def __init__(self): self.best, self.bad, self.state, self.epoch = float("inf"), 0, None, -1
        def step(self, v, e):
            if v < self.best * (1 - a.min_delta):
                self.best, self.bad, self.epoch = v, 0, e
                self.state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                self.bad += 1
            return bool(a.patience) and self.bad >= a.patience
        def restore(self):
            if a.patience and self.state is not None:
                model.load_state_dict(self.state)

    # stage 1: encoders, contrastive only
    opt = torch.optim.AdamW(model.encoders.parameters(), lr=a.lr, weight_decay=a.wd)
    t = time.time()
    r_stage, r_epoch = 1, -1
    if a.stage1_ckpt:
        model.load_state_dict(torch.load(a.stage1_ckpt, map_location=dev)["state_dict"]); r_stage, r_epoch = 2, -1
        print(f"[lad] stage 1 skipped, encoders from {a.stage1_ckpt}", flush=True)
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["state_dict"]); r_stage, r_epoch = ck["stage"], ck["epoch"]
        print(f"[lad] resumed {a.resume}: stage {r_stage}, epoch {r_epoch}", flush=True)
    stop1 = Stopper()
    e = r_epoch
    for e in range(r_epoch + 1 if r_stage == 1 else a.enc_epochs, a.enc_epochs):
        tau = temperature(min(e, tau_n - 1), tau_n, a.t0, a.t1)
        for b in batches():
            opt.zero_grad()
            z = model.encode(b)
            loss, g = pair_contrastive(z, tau)
            torch.autograd.backward([z[h] for h in hands], [g[h] for h in hands]); opt.step()
        if e % a.eval_every == 0 or e == a.enc_epochs - 1:
            with torch.no_grad():
                model.eval(); vl = pair_contrastive(model.encode(Xva_c), tau)[0]; model.train()
            r = {"stage": 1, "epoch": e, "tau": tau, "train_contrastive": loss, "val_contrastive": vl, "sec": time.time() - t}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True); run.metric(r)
            save("lad_cam_last.pt", stage=1, epoch=e)
            if e >= tau_n - 1 and stop1.step(vl, e):          # the loss is comparable only at the final temperature
                print(f"[lad] stage 1 early stop at epoch {e}: best val_contrastive {stop1.best:.5f} at epoch {stop1.epoch}", flush=True)
                break
    if r_stage == 1:
        # stage-1 result kept apart (lad_cam_last.pt is overwritten by stage 2): the last-epoch weights, to resume stage 1
        # with a larger --enc_epochs when the cap was reached while the loss was still falling (capped = True)
        capped = bool(a.patience) and stop1.bad < a.patience
        save("lad_cam_stage1_last.pt", stage=1, epoch=e, capped=capped)
        if capped:
            print(f"[lad] stage 1 reached the cap of {a.enc_epochs} epochs without early stop (best {stop1.best:.5f} at epoch "
                  f"{stop1.epoch}); resume with --resume lad_cam_stage1_last.pt --enc_epochs <larger>", flush=True)
        stop1.restore()
        save("lad_cam_stage1.pt", stage=1, epoch=stop1.epoch)     # the stage-1 result both stage-2 variants start from
    run.note(stage1_epochs=e + 1, stage1_best_epoch=stop1.epoch, stage1_best_val_contrastive=stop1.best)

    # stage 2: decoders (reconstruction) + encoders fine-tuned at the lower lr, L = L_recon + lam * L_contrastive
    if a.freeze_encoder:
        opt = torch.optim.AdamW(model.decoders.parameters(), lr=a.ft_lr, weight_decay=a.wd)
        model.encoders.eval()
    else:
        opt = torch.optim.AdamW([{"params": model.decoders.parameters(), "lr": a.ft_lr},
                                 {"params": model.encoders.parameters(), "lr": a.lr}], weight_decay=a.wd)
    extra = hooks.stage2_extra(a, model, hands, norm, dev)
    stop2 = Stopper()
    for e in range(r_epoch + 1 if r_stage == 2 else 0, a.dec_epochs):
        for b in batches():
            opt.zero_grad()
            if a.freeze_encoder:
                with torch.no_grad():
                    z = model.encode(b)
                rec = sum(F.mse_loss(model.decoders[h](z[h]), b[h]) for h in hands) / len(hands)
                rec.backward(); opt.step(); con = 0.0
                continue
            z = model.encode(b)
            rec = sum(F.mse_loss(model.decoders[h](z[h]), b[h]) for h in hands) / len(hands)
            con, g = pair_contrastive(z, a.t1)
            ex, info = extra() if extra else (0.0, {})
            # gradient of rec (+ a variant's extra loss) + lam * con
            torch.autograd.backward([rec + ex] + [z[h] for h in hands], [None] + [a.lam * g[h] for h in hands]); opt.step()
        if e % a.eval_every == 0 or e == a.dec_epochs - 1:
            cr = cross_recon(model, Xva, std)
            if a.freeze_encoder:
                model.encoders.eval()
            sr = {h: cr[f"{h}_to_{h}"] for h in hands}
            r = {"stage": 2, "epoch": e, "train_recon": float(rec), "train_contrastive": con,
                 **(info if not a.freeze_encoder else {}),
                 "val_SR_mean": float(np.mean(list(sr.values()))),
                 "val_CR_mean": float(np.mean([v for k, v in cr.items() if k.split("_to_")[0] != k.split("_to_")[1]])),
                 "sec": time.time() - t}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True); run.metric(r)
            save("lad_cam_last.pt", stage=2, epoch=e)
            if stop2.step(r["val_SR_mean"] + r["val_CR_mean"], e):
                print(f"[lad] stage 2 early stop at epoch {e}: best val SR+CR {stop2.best:.5f} at epoch {stop2.epoch}", flush=True)
                break
    stop2.restore()
    run.note(stage2_epochs=e + 1, stage2_best_epoch=stop2.epoch, stage2_best_val_SR_plus_CR=stop2.best)

    cr = cross_recon(model, Xva, std)
    json.dump(cr, open(f"{a.out}/cross_recon.json", "w"), indent=1)
    save("lad_cam.pt")
    print("[lad] saved", f"{a.out}/lad_cam.pt", flush=True)
    sr = [cr[f"{h}_to_{h}"] for h in hands]
    run.finish(val_SR_mean=float(np.mean(sr)), val_CR_mean=float(np.mean([v for k, v in cr.items() if k.split("_to_")[0] != k.split("_to_")[1]])))


if __name__ == "__main__":
    main()
