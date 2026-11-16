"""Ours, not the LAD paper (2026-11-16, user: "1번이랑 4번으로 진행"): LAD contrastive action model with a wider coverage
of hand poses, so the latent also serves task demonstrations and poses outside the retargeted grasp distribution.

Built on the reproduction ``dexverse_lad.train_cam`` (cross_dex/baselines/lad, robot_clip unchanged) through its Hooks:
  --norm limits   input normalisation by the URDF joint limits -> [-1, 1] (XL-VLA) instead of the per-joint mean / std of
                  the pairs, so poses outside the pair distribution stay in range ("4")
  --rand_w W      fine-tuned stage 2: + W * self reconstruction of joint configurations sampled uniformly in the limits,
                  fresh every step for every hand (XL-VLA's pose generator) ("1")
  --geo_w W       + W * XL-VLA cross-hand geometry on those random poses: hand A's pose decoded on hand B keeps A's
                  thumb->finger fingertip lines (lambda_dis * length MSE + lambda_dir * (1 - cos)), canonical-frame FK of
                  dexverse_xlvla ("1")

  cd /workspace/codes/dexverse/robot_clip && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.lad_cover --pairs <pairs.npy> --out <dir> --norm limits --rand_w 1 --geo_w 1 [train_cam options]
(every option of dexverse_lad.train_cam works the same; checkpoints load with the reproduction's code)
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("ROBOT_CLIP_ROOT", "/workspace/codes/dexverse/robot_clip"))
sys.path.insert(0, os.environ.get("DEXLATENT_ROOT", "/workspace/codes/dexverse/DexLatent"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dexverse_lad import train_cam as TC  # noqa: E402


def hand_models(pairs, hands, left=False):
    """dexverse_xlvla FK models of the pair file's hands (DexVerse URDFs, or dj_<hand> for DexJoCo-X) with their joint limits
    in the pair columns' order: {hand: {model, idx (model dof -> column), lo, hi}}."""
    from dexverse_xlvla import hands as H
    pm = json.load(open(pairs[:-4] + ".meta.json"))
    out = {}
    for h in hands:
        m = H.CanonicalHandModel(("dj_" if pm["bench"] == "dexjoco" else "") + h + ("_left" if left else ""))
        names = pm["joint_names"][h]
        if left and pm["bench"] == "dexverse":
            try:
                m.data_index(names)
            except KeyError:
                names = [x.replace("right_", "left_") for x in names]
        idx = m.data_index(names)
        lo, hi = torch.empty(len(idx)), torch.empty(len(idx))
        lo[idx], hi[idx] = m._lower, m._upper
        out[h] = {"model": m, "idx": torch.tensor(idx), "lo": lo, "hi": hi}
    return out


def fk_tips(fk, x):
    """Limits-normalised joints in the pair columns' order -> canonical fingertips [B, T, 3] (differentiable)."""
    return fk["model"].forward(x[:, fk["idx"].to(x.device)])


class CoverHooks(TC.Hooks):
    def add_args(self, ap):
        ap.add_argument("--norm", choices=["zscore", "limits"], default="limits")
        ap.add_argument("--rand_w", type=float, default=0.0)
        ap.add_argument("--rand_batch", type=int, default=4096)
        ap.add_argument("--geo_w", type=float, default=0.0)
        ap.add_argument("--lambda_dis", type=float, default=2000.0)   # XL-VLA / DexLatent TrainingConfig defaults
        ap.add_argument("--lambda_dir", type=float, default=5.0)

    def setup(self, a, raw, hands, norm):
        if (a.rand_w or a.geo_w) and a.norm != "limits":
            raise SystemExit("--rand_w / --geo_w sample in the normalised space: use --norm limits")
        self.fk = hand_models(a.pairs, hands, a.left) if (a.norm == "limits" or a.geo_w) else None
        if a.norm == "limits":                     # limits -> [-1, 1]: mean = centre, std = half range (stored like z-score stats)
            norm = {h: {"mean": ((self.fk[h]["hi"] + self.fk[h]["lo"]) / 2).numpy(),
                        "std": ((self.fk[h]["hi"] - self.fk[h]["lo"]) / 2).numpy() + 1e-8} for h in hands}
        return norm

    def stage2_extra(self, a, model, hands, norm, dev):
        if not (a.rand_w or a.geo_w):
            return None
        gen = torch.Generator(device=dev); gen.manual_seed(a.seed + 7)

        def extra():
            R = {h: torch.rand(a.rand_batch, len(norm[h]["mean"]), generator=gen, device=dev) * 2 - 1 for h in hands}
            zr = model.encode(R)
            out, lr, lg = 0.0, 0.0, 0.0
            if a.rand_w:
                lr = sum(F.mse_loss(model.decoders[h](zr[h]), R[h]) for h in hands) / len(hands); out = out + a.rand_w * lr
            if a.geo_w:
                terms = []
                for s in hands:
                    ts = fk_tips(self.fk[s], R[s]).detach()
                    for t in hands:
                        if t == s:
                            continue
                        tt = fk_tips(self.fk[t], model.decoders[t](zr[s]).clamp(-1, 1))
                        k = min(ts.shape[1], tt.shape[1])
                        ls, lt = ts[:, 1:k] - ts[:, :1], tt[:, 1:k] - tt[:, :1]          # thumb -> finger lines
                        terms.append(a.lambda_dis * F.mse_loss(lt.norm(dim=-1), ls.norm(dim=-1))
                                     + a.lambda_dir * (1 - F.cosine_similarity(lt, ls, dim=-1)).mean())
                lg = sum(terms) / len(terms); out = out + a.geo_w * lg
            return out, {"train_rand_recon": float(lr), "train_geo": float(lg)}
        return extra


if __name__ == "__main__":
    TC.main(hooks=CoverHooks())
