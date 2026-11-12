"""Cross-embodiment chunk-level latent action space (per-hand encoder / decoder, one shared latent).

A finger-target chunk [H, n_h] of hand h is encoded to K latent tokens [K, d] (temporal compression H/K, default
16 -> 4) and decoded back by any hand's decoder. Losses (after LAD, Bauer et al. 2025, arXiv:2506.14608, adapted to
exactly paired chunks):
  * contrastive: symmetric InfoNCE between the flattened, L2-normalised latents of the same chunk on two hands
    (positives = same batch index, negatives = rest of the batch), averaged over hand pairs, temperature schedule;
  * reconstruction: decode(encode_h(x_h)) -> x_h;
  * cross-reconstruction: decode_g(encode_h(x_h)) -> x_g (LAD uses it only as a metric; our pairs are exact, so it is
    trained: it enforces that one latent means the same motion on every hand).
Inputs are z-scored per hand joint (dataset train statistics).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkEncoder(nn.Module):
    def __init__(self, n_in: int, H: int, K: int, d: int, width: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        assert H % K == 0
        self.H, self.K = H, K
        self.inp = nn.Linear(n_in, width)
        self.pos = nn.Parameter(torch.zeros(1, H, width))
        nn.init.normal_(self.pos, std=0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True,
                                                                   norm_first=True), layers)
        self.down = nn.Conv1d(width, width, kernel_size=H // K, stride=H // K)   # H -> K tokens
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
        self.logvar = None

    def forward(self, x):                      # [B, H, n] -> [B, K, d] (VAE: posterior mean)
        return self.features(x)[0]

    def features(self, x):
        h = self.tf(self.inp(x) + self.pos)
        h = self.down(h.transpose(1, 2)).transpose(1, 2)
        return self.out(h), (self.logvar(h) if self.logvar is not None else None)


class ChunkDecoder(nn.Module):
    def __init__(self, n_out: int, H: int, K: int, d: int, width: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.H, self.K = H, K
        self.inp = nn.Linear(d, width)
        self.pos = nn.Parameter(torch.zeros(1, H, width))
        nn.init.normal_(self.pos, std=0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True,
                                                                   norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def forward(self, z):                      # [B, K, d] -> [B, H, n]
        h = self.inp(z).repeat_interleave(self.H // self.K, dim=1) + self.pos
        return self.out(self.tf(h))


class ChunkLatent(nn.Module):
    """vae=False: deterministic auto-encoder (LAD). vae=True: Gaussian posterior per token (mean / log-variance head),
    sampled z for decoding in training, KL to N(0, I) weighted by beta (XL-VLA paper beta 1e-5); encode() returns the
    mean (contrastive term, evaluation, policy data)."""

    def __init__(self, dofs: dict[str, int], H: int = 16, K: int = 4, d: int = 32, width: int = 128, layers: int = 2,
                 vae: bool = False):
        super().__init__()
        self.hands, self.H, self.K, self.d, self.vae = list(dofs), H, K, d, vae
        self.enc = nn.ModuleDict({h: ChunkEncoder(n, H, K, d, width, layers) for h, n in dofs.items()})
        self.dec = nn.ModuleDict({h: ChunkDecoder(n, H, K, d, width, layers) for h, n in dofs.items()})
        if vae:
            for e in self.enc.values():
                e.logvar = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))

    def encode(self, h: str, x):
        return self.enc[h](x)

    def posterior(self, h: str, x):
        return self.enc[h].features(x)

    def decode(self, h: str, z):
        return self.dec[h](z)


def info_nce(za, zb, tau: float):
    """Symmetric InfoNCE on flattened, L2-normalised latents [B, K, d]."""
    a = F.normalize(za.flatten(1), dim=-1)
    b = F.normalize(zb.flatten(1), dim=-1)
    logits = a @ b.t() / tau
    y = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


def tau_at(step: int, total: int, t0: float = 0.25, t1: float = 0.16) -> float:
    """Exponential decay t0 -> t1 over the run (LAD Table II)."""
    return t0 * math.exp(math.log(t1 / t0) * min(step / max(total, 1), 1.0))


def _rec(y, x, w_vel: float):
    """MSE + w_vel * MSE of step differences (decoded chunks otherwise jitter: jerk 5-31x the 60 Hz data, 2026-11-12)."""
    l = F.mse_loss(y, x)
    if w_vel and y.shape[1] > 1:
        l = l + w_vel * F.mse_loss(y[:, 1:] - y[:, :-1], x[:, 1:] - x[:, :-1])
    return l


def losses(model: ChunkLatent, x: dict[str, torch.Tensor], tau: float, stage: int, w_contr: float = 0.1,
           w_cross: float = 1.0, w_vel: float = 0.0, beta: float = 1e-5) -> dict[str, torch.Tensor]:
    """stage 1: contrastive only (encoders); stage 2: recon + cross recon (+ w_vel velocity term) + w_contr * contrastive."""
    post = {h: model.posterior(h, x[h]) for h in model.hands}
    z = {h: post[h][0] for h in model.hands}                       # mean (AE: the latent)
    pairs = [(a, b) for i, a in enumerate(model.hands) for b in model.hands[i + 1:]]
    out = {"contr": sum(info_nce(z[a], z[b], tau) for a, b in pairs) / len(pairs)}
    if stage == 1:
        out["loss"] = out["contr"]
        return out
    zs = z
    if model.vae:                                                  # reparameterised sample for decoding
        zs = {h: z[h] + torch.randn_like(z[h]) * torch.exp(0.5 * post[h][1]) for h in model.hands}
    rec = {h: _rec(model.decode(h, zs[h]), x[h], w_vel) for h in model.hands}
    cross = [_rec(model.decode(g, zs[h]), x[g], w_vel) for h in model.hands for g in model.hands if g != h]
    out["recon"] = sum(rec.values()) / len(rec)
    out["cross"] = sum(cross) / len(cross)
    out["loss"] = out["recon"] + w_cross * out["cross"] + w_contr * out["contr"]
    if model.vae:
        kl = [(-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(-1).mean() for mu, lv in post.values()]
        out["kl"] = sum(kl) / len(kl)
        out["loss"] = out["loss"] + beta * out["kl"]
    return out


def load_model(path: str, device="cpu"):
    """Checkpoint of train.py -> (model in eval mode, ckpt dict with norm / hands / H)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ChunkLatent(ck["dofs"], ck["H"], a["K"], a["d"], a["width"], a["layers"], vae=bool(a.get("vae", 0)))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


class HandCodec:
    """Encode / decode finger-target chunks of one hand in joint units (z-score inside)."""

    def __init__(self, model: ChunkLatent, ck: dict, hand: str, device="cpu"):
        self.m, self.h, self.dev = model, hand, device
        self.mean = torch.tensor(ck["norm"][hand]["mean"], dtype=torch.float32, device=device)
        self.std = torch.tensor(ck["norm"][hand]["std"], dtype=torch.float32, device=device).clamp_min(1e-3)
        self.H, self.K, self.d = model.H, model.K, model.d

    @torch.no_grad()
    def encode(self, q):                        # [B, H, n] joint angles -> [B, K, d]
        q = torch.as_tensor(q, dtype=torch.float32, device=self.dev)
        return self.m.encode(self.h, (q - self.mean) / self.std)

    @torch.no_grad()
    def decode(self, z):                        # [B, K, d] -> [B, H, n] joint angles
        z = torch.as_tensor(z, dtype=torch.float32, device=self.dev)
        return self.m.decode(self.h, z) * self.std + self.mean
