"""Adapter example: an XL-VLA hand latent checkpoint (dexverse_xlvla, dexlatent env). Same numbers as decode_xlvla."""
import torch
from dexverse_xlvla.eval_latent import dec, enc, load_eval
from dexverse_lad.eval_space import model_name


class XLVLA:
    def __init__(self, ckpt, hands, joint_names, side, bench):
        self.tr = load_eval(ckpt)
        self.mn = {h: model_name(h, side, bench) for h in hands}
        self.M = {h: self.tr.hand_models[self.mn[h]] for h in hands}
        self.idx = {h: self.M[h].data_index(joint_names[h]) for h in hands}   # model dof d = column idx[d]

    def encode(self, h, q):
        return enc(self.tr, self.mn[h], self.M[h].angles_to_normalized(q[:, self.idx[h]]).clamp(-1, 1))

    def decode(self, h, z):
        M = self.M[h]; qn = dec(self.tr, self.mn[h], z).clamp(-1, 1)
        ang = (qn + 1) * 0.5 * (M._upper - M._lower) + M._lower
        q = torch.zeros_like(ang); q[:, self.idx[h]] = ang
        return q


def load(ckpt, hands, joint_names, side, bench):
    return XLVLA(ckpt, hands, joint_names, side, bench)
