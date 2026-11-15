"""On-the-fly paired chunks from a keyframe store (build_keyframes.py): per sampled grasp, every hand's joints follow the
same min-jerk timing through its own 3 keyframes (hold 0.1 s, pregrasp->grasp over D = closing time / 0.507 with the
closing time drawn from the type's human quantiles p10/p50/p90 (uniform in [0.1, 0.9] quantile, piecewise linear),
grasp->squeeze 0.2 s, hold 0.3 s), and a random H-step window is cut. All on the GPU (a few tensor ops per batch)."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

MINJERK_10_90 = 0.507


def _mj(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class KeyframeSampler:
    def __init__(self, root: str, hands: list[str], mean: dict, std: dict, H: int, fps: float, device, split: int = 0,
                 src: dict | None = None, order: dict | None = None, stretch: tuple = (1.0, 1.0)):
        """src: training hand -> keyframe-store hand (e.g. DexJoCo wuji -> wuji_dj); order: training hand -> joint index
        list into the store hand's joints (benchmark actuator order); stretch: closing time x U(lo, hi) (task speed)."""
        z = np.load(os.path.join(root, "kf.npz"))
        meta = json.load(open(os.path.join(root, "meta.json")))
        sel = z["split"] == split
        self.hands, self.H, self.fps, self.dev = hands, H, fps, device
        src, order = src or {}, order or {}
        self.stretch = stretch

        def load(h):
            q = z[f"q_{src.get(h, h)}"][sel]
            return q[..., order[h]] if h in order else q
        self.Q = {h: ((torch.from_numpy(load(h)).to(device) - mean[h]) / std[h]) for h in hands}   # [N, 3, n]
        self.feix = torch.from_numpy(z["feix"][sel].astype(np.int64)).to(device)
        types = sorted(int(g) for g in meta["close_q"])
        lut = torch.zeros(max(types) + 1, 3)
        for g in types:
            lut[g] = torch.tensor(meta["close_q"][str(g)])
        self.close_q = lut.to(device)
        self.N = len(self.feix)
        # type-balanced sampling weights
        cnt = torch.bincount(self.feix, minlength=max(types) + 1).float()
        self.w = (1.0 / cnt[self.feix]).clamp(max=1.0)

    def sample(self, B: int, balance: bool = True, gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        dev, H, fps = self.dev, self.H, self.fps
        i = torch.multinomial(self.w, B, replacement=True, generator=gen) if balance else \
            torch.randint(self.N, (B,), device=dev, generator=gen)
        q = self.close_q[self.feix[i]]                                               # [B, 3] p10 p50 p90
        u = torch.rand(B, device=dev, generator=gen) * 0.8 + 0.1
        close = torch.where(u < 0.5, q[:, 0] + (u - 0.1) / 0.4 * (q[:, 1] - q[:, 0]), q[:, 1] + (u - 0.5) / 0.4 * (q[:, 2] - q[:, 1]))
        lo, hi = self.stretch
        if hi > lo or lo != 1.0:
            close = close * (lo + (hi - lo) * torch.rand(B, device=dev, generator=gen))
        n0 = round(0.1 * fps)
        n1 = torch.clamp((close / MINJERK_10_90 * fps).round(), min=2).long()        # frames of pregrasp->grasp
        n2, n3 = round(0.2 * fps), round(0.3 * fps)
        L = n0 + n1 + n2 + n3
        t0 = (torch.rand(B, device=dev, generator=gen) * (L - H + 1).float()).floor().long().clamp(min=0)
        f = t0[:, None] + torch.arange(H, device=dev)[None]                          # [B, H] frame index
        s1 = ((f - n0 + 1).float() / n1[:, None].float()).clamp(0, 1)
        s2 = ((f - n0 - n1[:, None] + 1).float() / n2).clamp(0, 1)
        w1, w2 = _mj(s1)[..., None], _mj(s2)[..., None]                             # [B, H, 1]
        out = {}
        for h in self.hands:
            k = self.Q[h][i]                                                          # [B, 3, n]
            a, b, c = k[:, None, 0], k[:, None, 1], k[:, None, 2]
            out[h] = a + w1 * (b - a) + w2 * (c - b)                                  # w2 = 0 before the grasp segment ends
        return out
