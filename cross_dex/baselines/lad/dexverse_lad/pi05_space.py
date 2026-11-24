"""LAD latent as the finger action space of a policy: encode / decode between the finger vector the DexVerse policy code
uses (hand-model DoF order, joint range -> [-1, 1]; dexverse_xlvla CanonicalHandModel) and the LAD embedding (16).

The contrastive action model (dexverse_lad.train_cam, authors' robot_clip.model.RobotCLIP unchanged) was trained on finger
targets in ENV ACTION order and joint units, z-scored per dimension. So

  encode: normalised model-order fingers -> joint angles -> env action order -> z-score -> encoder_h -> z (16)
  decode: z -> decoder_h -> un-z-score -> joint angles (env action order) -> model order -> [-1, 1]

Joint names of the LAD inputs: ``hands.json`` next to the checkpoint (HF Dexverse-CSI/shared-action-spaces layout) or
``finger_joint_names.json`` (pairs meta), or pass ``names``. Encoders / decoders stay frozen (paper Sec. III-C).
"""
import json
import os
import sys
import types

import torch


def _robot_clip_model():
    """robot_clip.model imports omegaconf only for a type hint; the policy envs (dexlatent) do not have it."""
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        stub = types.ModuleType("omegaconf")
        stub.DictConfig = dict
        sys.modules["omegaconf"] = stub
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from robot_clip.model import RobotCLIP
    return RobotCLIP


class _NS(types.SimpleNamespace):
    def items(self):
        return self.__dict__.items()


def _cfg(d):
    """plain dict (saved OmegaConf container) -> attribute access as RobotCLIP expects"""
    if isinstance(d, dict):
        return _NS(**{k: _cfg(v) for k, v in d.items()})
    return d


class LadSpace:
    def __init__(self, ckpt: str, hand_models: dict, names: str | None = None):
        """hand_models: {hand: CanonicalHandModel} (dof_joints, _lower, _upper, hand_name); names: finger_joint_names json"""
        p = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = _robot_clip_model()(_cfg(p["config"]))
        self.model.load_state_dict(p["state_dict"])
        self.model.eval()
        self.dim = int(p["config"]["model"]["embedding_dim"])
        d = os.path.dirname(os.path.abspath(ckpt))
        names = names or next(f for f in (f"{d}/hands.json", f"{d}/finger_joint_names.json") if os.path.exists(f))
        jn = json.load(open(names))
        if "hands" in jn:      # shared-action-spaces hands.json: {"side", "hands": {hand: {"n", "joint_names_<side>": [...]}}}
            jn = {h: v[f"joint_names_{jn.get('side', 'right')}"] for h, v in jn["hands"].items() if f"joint_names_{jn.get('side', 'right')}" in v}
        jn = jn.get("finger_joint_names", jn)
        from dexverse_xlvla.hands import HANDS
        self.mean, self.std, self.perm, self.inv, self.m = {}, {}, {}, {}, hand_models
        for h in p["hands"]:
            if h not in hand_models:
                continue
            m = hand_models[h]
            alias = HANDS[m.hand_name].data_name            # dataset (env) joint name -> URDF joint name
            urdf = [alias.get(n, n) for n in jn[h]]
            if sorted(urdf) != sorted(m.dof_joints):
                raise ValueError(f"{h}: LAD joints {jn[h]} != hand model DoFs {list(m.dof_joints)}")
            self.perm[h] = torch.tensor([list(m.dof_joints).index(j) for j in urdf])      # LAD order <- model order
            self.inv[h] = torch.tensor([urdf.index(j) for j in m.dof_joints])             # model order <- LAD order
            self.mean[h] = torch.tensor(p["normalization"][h]["mean"], dtype=torch.float32)
            self.std[h] = torch.tensor(p["normalization"][h]["std"], dtype=torch.float32)

    def angles(self, h, qn):
        m = self.m[h]
        return (qn.clamp(-1, 1) + 1) * 0.5 * (m._upper - m._lower) + m._lower

    @torch.no_grad()
    def encode(self, h: str, qn: torch.Tensor) -> torch.Tensor:
        x = self.angles(h, qn)[:, self.perm[h]]
        return self.model.encoders[h]((x - self.mean[h]) / self.std[h])

    @torch.no_grad()
    def decode(self, h: str, z: torch.Tensor) -> torch.Tensor:
        x = self.model.decoders[h](z) * self.std[h] + self.mean[h]
        return self.m[h].angles_to_normalized(x[:, self.inv[h]]).clamp(-1, 1)
