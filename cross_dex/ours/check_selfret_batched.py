"""Equality + speed check of xlvla_local_selfret's batched path (G3, --sr_batched 1) against the per-direction path.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.check_selfret_batched [--refine_iters 10]

Same weights, same uniform poses: total self-retargeting loss, per-direction residual / pass %, gradients; then
seconds per call of each path.
"""
from __future__ import annotations

import argparse
import copy
import time

import torch

from cross_dex.ours import check_fast as CF
from cross_dex.ours import fast_xl as FX
from cross_dex.ours import xlvla_local_selfret as SR


def build(a, batched):
    tr = CF.make_trainer(SR.SelfRetargetTrainer)
    FX.make_fast(tr)
    tr.fast = True
    tr.enable_local(0.0, 1.0, 1024)
    ns = argparse.Namespace(w_rt=1.0, w_cyc=1.0, rt_scale=1.0, refine_iters=a.refine_iters, refine_until=0,
                            refine_lr=0.05, w_ref=1.0, cyc_warmup=1, sr_batch=1024, res_min_mm=15.0, res_cap_mm=30.0,
                            res_ema=0.99, sr_log_every=10 ** 9, sr_batched=batched)
    tr.enable_selfret(ns)
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine_iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    to, tb = build(a, 0), build(a, 1)
    tb.autoencoders.load_state_dict(to.autoencoders.state_dict())
    torch.manual_seed(3)
    qs_of = {h: torch.empty(1024, to.hand_models[h].dof_count(), device="cuda").uniform_(-1, 1) for h in to.hand_names}
    for step in range(2):                                      # 2 calls: first (median init) and EMA update
        for t in (to, tb):
            t.autoencoders.zero_grad(); t._sr_loss = 0.0
            t._selfret_backward(qs_of)
        g = max(float((p.grad - q.grad).abs().max()) for p, q in zip(to.autoencoders.parameters(), tb.autoencoders.parameters())
                if p.grad is not None)
        gn = max(float(p.grad.abs().max()) for p in to.autoencoders.parameters() if p.grad is not None)
        ro, po = to.sr_detail
        rb, pb = tb.sr_detail
        print(f"call {step}: loss {to._sr_loss:.6g}/{tb._sr_loss:.6g}  grad max|diff| {g:.2e} (max|grad| {gn:.2e})  "
              f"res max|diff| {max(abs(ro[k] - rb[k]) for k in ro):.2e} mm  pass max|diff| {max(abs(po[k] - pb[k]) for k in po):.2e} %  "
              f"metrics " + " ".join(f"{k} {to.extra_metrics[k]:.5g}/{tb.extra_metrics[k]:.5g}" for k in
                                     ("sr_rt", "sr_cyc", "sr_res_med_mm", "sr_dec_res_med_mm", "sr_pass_pct")), flush=True)
    for name, t in (("per-direction", to), ("batched", tb)):
        for _ in range(2):
            t.autoencoders.zero_grad(); t._selfret_backward()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(a.reps):
            t.autoencoders.zero_grad(); t._selfret_backward()
        torch.cuda.synchronize()
        print(f"speed {name}: {(time.time() - t0) / a.reps:.3f} s/call (refine {a.refine_iters} it)", flush=True)


if __name__ == "__main__":
    main()
