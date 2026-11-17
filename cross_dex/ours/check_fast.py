"""Equality + speed check of cross_dex.ours.fast_xl against the original code (run before using --fast 1).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.check_fast \
      [--pairs <uniform_pairs.npz>] [--steps 20]

1 FK: fingertips / wrist of FastCanonicalHandModel vs CanonicalHandModel, values and input gradients, every hand.
2 XL step: CrossEmbodimentTrainer.step (original) vs FastXLMixin.step, same weights and RNG: losses + updated weights.
3 local loss: LocalTrainer._local_loss vs fast_local_loss, same weights and RNG: value + gradients.
4 speed: seconds per training step of the full ours config (XL + local + pair buffer), original vs fast.
5 torch.compile of the fast FK (option 2): equality and speed, if it compiles.
"""
from __future__ import annotations

import argparse
import copy
import time

import torch

import HandLatent.model as M
from cross_dex.ours import fast_xl as FX
from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import hands as H
from dexverse_xlvla import train_latent as TL


def maxdiff(a, b):
    return float((a - b).abs().max())


def make_trainer(cls):
    H.install()
    M.HandAutoencoder = TL.SampledHandAutoencoder
    cfg = M.TrainingConfig(num_steps=10, checkpoint_interval=0, checkpoint_dir="/workspace/.tmp/check_fast_ck", arm_dof=0,
                           pinch_offset=(0.0, 0.0, 0.0), pinch_sampling_probability=0.0, lambda_kl=1e-5)
    return cls(list(H.HAND_NAMES), cfg)


LT = XLL.LocalTrainer                      # FastXLMixin inside; fast=False -> original step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    print("== 1 FK")
    for h in H.HAND_NAMES:
        o, f = H.CanonicalHandModel(h), FX.FastCanonicalHandModel(h)
        q = torch.empty(4096, o.dof_count(), device=dev).uniform_(-1.2, 1.2).requires_grad_(True)
        to, wo, _ = o._forward_internal(q)
        go = torch.autograd.grad(to.sum() + wo.sum(), q)[0]
        tf, wf, _ = f._forward_internal(q)
        gf = torch.autograd.grad(tf.sum() + wf.sum(), q)[0]
        print(f"{h:8s} tips {maxdiff(to, tf):.2e} m  wrist {maxdiff(wo, wf):.2e}  grad {maxdiff(go, gf):.2e}  "
              f"canon {maxdiff(o.canon, f.canon):.2e}", flush=True)

    print("== 2 XL step (original vs fast, same weights / RNG)")
    t_o = make_trainer(LT)
    t_f = make_trainer(LT)
    t_f.autoencoders.load_state_dict(t_o.autoencoders.state_dict())
    FX.make_fast(t_f)
    t_f.fast = True
    for k in range(3):
        torch.manual_seed(100 + k); mo = t_o.step()
        torch.manual_seed(100 + k); mf = t_f.step()
        print(f"step {k}: " + " ".join(f"{n} {mo[n]:.6g}/{mf[n]:.6g}" for n in mo), flush=True)
    w = max(maxdiff(p, q) for p, q in zip(t_o.autoencoders.parameters(), t_f.autoencoders.parameters()))
    print(f"weights after 3 steps: max |diff| {w:.2e}", flush=True)

    print("== 3 local loss")
    for t in (t_o, t_f):
        t.w_local, t.local_deg, t.local_batch = 1.0, 1.0, 1024
    for t in (t_o, t_f):
        t.autoencoders.zero_grad()
    torch.manual_seed(7); lo = t_o._local_loss(); lo.backward()
    torch.manual_seed(7); lf = FX.fast_local_loss(t_f); lf.backward()
    g = max(maxdiff(p.grad, q.grad) for p, q in zip(t_o.autoencoders.parameters(), t_f.autoencoders.parameters())
            if p.grad is not None)
    print(f"local loss {float(lo):.6g} / {float(lf):.6g}, grad max |diff| {g:.2e}", flush=True)

    if a.pairs:
        print("== 4 speed, full ours config (XL + local + pair buffer, no refresh)")
        from cross_dex.ours import xlvla_local_pair as XP
        ns = argparse.Namespace(pairs=a.pairs, w_pair=1.0, pair_batch=16384, pair_tau=0.16, pair_max_res_mm=15.0,
                                pair_buffer=200000, refresh_every=0, refresh_n=0, refresh_iters=0, refresh_starts=1)
        for fast in (0, 1):
            tr = make_trainer(XP.PairTrainer)
            if fast:
                FX.make_fast(tr); tr.fast = True; tr._local_loss = lambda tr=tr: FX.fast_local_loss(tr)
            tr.enable_local(1.0, 1.0, 1024)
            tr.enable_pair(ns)
            for _ in range(3):
                tr.step()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(a.steps):
                tr.step()
            torch.cuda.synchronize()
            print(f"fast={fast}: {(time.time() - t0) / a.steps:.3f} s/step", flush=True)
            del tr

    print("== 5 torch.compile of the fast FK")
    f = FX.FastCanonicalHandModel("shadow")
    q = torch.empty(1024, f.dof_count(), device=dev).uniform_(-1, 1)
    try:
        comp = torch.compile(f._raw, dynamic=False)
        r1 = comp(q)[0]; r0 = f._raw(q)[0]
        for fn, name in ((f._raw, "eager"), (comp, "compiled")):
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(50):
                fn(q)
            torch.cuda.synchronize()
            print(f"{name}: {(time.time() - t0) / 50 * 1000:.2f} ms per FK", flush=True)
        print(f"compiled vs eager max |diff| {maxdiff(r0, r1):.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile failed: {type(e).__name__}: {str(e)[:300]}")
    o = H.CanonicalHandModel("shadow")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50):
        o._forward_internal(q)
    torch.cuda.synchronize()
    print(f"original: {(time.time() - t0) / 50 * 1000:.2f} ms per FK")


if __name__ == "__main__":
    main()
