"""Decode / latent dump of ANY shared action space on the fixed evaluation sets, through a small adapter (2026-11-19).
Writes the same files as dexverse_lad.eval_space decode / decode_xlvla, so `eval_space metrics` and pair_metrics.py
measure it exactly like the spaces in the artifact.

  python run_space.py decode --adapter <adapter.py> --ckpt <path> --name <name> --out <eval set dir>  -> decoded_<name>.npz
  python run_space.py zdump  --adapter <adapter.py> --ckpt <path> --name <name> --out <eval set dir> --zout <file.npz>

Adapter = a python file with
  def load(ckpt: str, hands: list[str], joint_names: dict[str, list[str]], side: str, bench: str) -> model
  model.encode(hand: str, q: torch.FloatTensor [N, n_joints]) -> torch.FloatTensor [N, d]
  model.decode(hand: str, z: torch.FloatTensor [N, d])        -> torch.FloatTensor [N, n_joints]
q = finger joint angles in rad, columns in the order joint_names[hand] (stored in inputs.npz "names"), no wrist.
Hands of DexVerse right: shadow allegro inspire leap sharpa wuji; DexJoCo-X right: 7 dj hands (see inputs.npz "hands").
Example: adapter_xlvla.py.
"""
import argparse, importlib.util, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baselines", "lad"))
from dexverse_lad import eval_space as E  # noqa: E402


def load_adapter(path, ckpt, I):
    spec = importlib.util.spec_from_file_location("space_adapter", path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hands = [str(h) for h in I["hands"]]
    return mod.load(ckpt, hands, json.loads(str(I["names"])), str(I["side"]), E.bench_of(I)), hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["decode", "zdump"])
    ap.add_argument("--adapter", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--zout", default="")
    a = ap.parse_args()
    I, kinds = E.load_inputs(a.out)
    assert E.chunk_H(I) == 1, "chunk evaluation sets are not supported here"
    model, hands = load_adapter(a.adapter, a.ckpt, I)
    z_of = lambda s, kind: model.encode(s, torch.tensor(I[f"{kind}_{s}"], dtype=torch.float32))
    dec = lambda t, z: model.decode(t, z).detach().float().cpu().numpy()
    with torch.no_grad():
        if a.step == "zdump":
            Z = {f"{k}_{s}": z_of(s, k).float().cpu().numpy() for s in hands for k in kinds if len(I[f"{k}_{s}"])}
            np.savez(a.zout, hands=np.array(hands), **Z)
            print("zdump", a.name, {k: v.shape for k, v in Z.items()}, flush=True)
            return
        out = {}
        for kind in kinds:
            for s in hands:
                if not len(I[f"{kind}_{s}"]):
                    continue
                z = z_of(s, kind)
                for t in hands:                                     # every target hand (s == t = self reconstruction)
                    out[f"{kind}_{s}_to_{t}"] = dec(t, z)
        E.extras(z_of, dec, hands, I, out)
    np.savez(f"{a.out}/decoded_{a.name}.npz", ckpt=a.ckpt, **out)
    print("decoded", a.name, a.ckpt, flush=True)


if __name__ == "__main__":
    main()
