"""Space-metric decode for the xhand finger-latent codecs (Dexverse-CSI/xhand-finger-latent: E2*, codec v2, codec v3),
same output format as dexverse_lad.eval_space.decode (decoded_<name>.npz), so `eval_space metrics` scores them unchanged.

The codec code is not in the DexVerse copy (arm_pads.py / model.py / metadata/ missing), so the model is rebuilt from the
state_dict and the joint convention was verified on the evaluation inputs (2026-11-18, amp7 dv_right teleop frames):
  - per hand: encoder MLP [dof,64,128,64] -> mean Linear(64,32); decoder MLP [32,64,128,64,dof] -> tanh; ReLU between layers
    (ReLU+tanh: self rec MSE 0.0006-0.0045 vs the checkpoint's validation rec 0.0021; other activations 2-10x worse)
  - joints in the simulator joint-name order (= eval input columns), normalized to [-1,1] by the DexVerse URDF limits
    (cross-hand thumb-finger line direction wuji/allegro/leap -> shadow/sharpa: 9-15 deg vs 20-76 deg for URDF order)
  - z = encoder mean (as in the policies' encode)

  cd DexLatent && python -m dexverse_xhand_eval --out <eval dir> --ckpt e2star=<best.pt> ...   (dexlatent env)
"""
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mlp(s):
    L = []
    for i in range(len(s) - 1):
        L.append(nn.Linear(s[i], s[i + 1]))
        if i < len(s) - 2:
            L.append(nn.ReLU())
    return nn.Sequential(*L)


def load_codec(path, hands):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, have = ck["state_dict"], ck["config"]["hands"].split(",")
    assert set(hands) <= set(have), (have, hands)
    net = {}
    for h in hands:
        dof = ck["config"]["hand_dofs"][h]
        e, m, d = mlp([dof, 64, 128, 64]), nn.Linear(64, 32), mlp([32, 64, 128, 64, dof])
        for mod, p in [(e, "encoder"), (m, "mean"), (d, "decoder")]:
            mod.load_state_dict({k[len(h) + len(p) + 2:]: v for k, v in sd.items() if k.startswith(f"{h}.{p}.")})
        net[h] = (e.eval(), m.eval(), d.eval())
    return net


def main():
    from dexverse_lad.eval_space import load_inputs, extras, chunk_H, unchunk
    from dexverse_xlvla import hands as H
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    a = ap.parse_args()
    I, kinds = load_inputs(a.out)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"]))
    M = {h: H.CanonicalHandModel(h) for h in hands}
    idx = {h: np.array(M[h].data_index(names[h])) for h in hands}
    lo = {h: np.zeros(len(idx[h]), np.float32) for h in hands}; hi = {h: np.zeros(len(idx[h]), np.float32) for h in hands}
    for h in hands:                                                   # URDF limits in data (input column) order
        lo[h][idx[h]] = M[h]._lower.numpy(); hi[h][idx[h]] = M[h]._upper.numpy()
    Hc = chunk_H(I)
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        net = load_codec(path, hands)

        def z_of(s, kind):
            q = I[f"{kind}_{s}"].reshape(-1, I[f"{kind}_{s}"].shape[-1])
            qn = torch.tensor(2 * (q - lo[s]) / (hi[s] - lo[s]) - 1, dtype=torch.float32).clamp(-1, 1)
            e, m, _ = net[s]
            return m(e(qn))

        def dec_t(t, z):
            y = torch.tanh(net[t][2](z)).numpy()
            return unchunk((y + 1) * 0.5 * (hi[t] - lo[t]) + lo[t], Hc)
        out = {}
        with torch.no_grad():
            for kind in kinds:
                for s in hands:
                    if not len(I[f"{kind}_{s}"]):
                        continue
                    z = z_of(s, kind)
                    for t in hands:
                        out[f"{kind}_{s}_to_{t}"] = dec_t(t, z)
            extras(z_of, dec_t, hands, I, out)
        np.savez(f"{a.out}/decoded_{name}.npz", ckpt=path, **out)
        print("decoded", name, path, flush=True)


if __name__ == "__main__":
    main()
