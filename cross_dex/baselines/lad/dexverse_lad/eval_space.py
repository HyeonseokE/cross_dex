"""Adopted latent-space metrics 1-3 (experiments/a4/RSS2027/README.md "공유 잠재 공간 평가 메트릭") for LAD models on DexVerse.

  1  teleop self reconstruction   teleop finger actions -> encode -> decode on the same hand: joint error (deg), fingertip error (mm)
  2  cross reconstruction         held-out retargeting pair rows: latent of hand A decoded on hand B vs hand B's retargeted
                                  joints: joint error (deg), fingertip error (mm), mean over A != B
  3  cross geometric consistency  teleop frames: thumb-finger fingertip line of hand A vs of hand B decoded from A's latent:
                                  direction (deg), length (mm), mean over A != B (same definition as dexverse_xlvla.eval_latent.cross)

Three steps, because the LAD model (robot_clip, omegaconf) and the hand FK models (DexLatent, urdf_parser_py) live in
different environments (no package is added to either):
  prep     (dexlatent env)  teleop frames of every hand in the LAD joint order + the pair rows        -> inputs.npz
  decode   (lad env)        per checkpoint: self / cross decodes of the teleop frames and pair rows   -> decoded_<name>.npz
  metrics  (dexlatent env)  FK and the numbers                                                         -> space_<name>.json, summary

  cd DexLatent && python -m dexverse_lad.eval_space prep --pairs_sub <dexverse_val20k.npz> --side right --out <dir>   (PYTHONPATH=robot_clip)
  cd robot_clip && python -m dexverse_lad.eval_space decode --out <dir> --ckpt frozen=<lad_cam.pt> --ckpt ft=<lad_cam.pt>
  cd DexLatent && python -m dexverse_lad.eval_space decode_xlvla --out <dir> --ckpt xlvla=<checkpoint_epoch_N.pt>   (optional, comparison)
  cd DexLatent && python -m dexverse_lad.eval_space metrics --out <dir>
"""
import argparse, glob, json, os

import numpy as np
import torch

TELEOP = os.environ.get("XLVLA_TELEOP_ROOT", "/workspace/datasets/dexverse/teleop_lerobot")


def model_name(h, side):
    return h if side == "right" else f"{h}_left"


def prep(a):
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.check_hands import load_actions
    P = np.load(a.pairs_sub, allow_pickle=True)
    names = json.loads(str(P["names"]))
    hands = [str(h) for h in P["hands"]]
    rng = np.random.default_rng(1000)
    out, info = {}, {}
    for h in hands:
        m = H.CanonicalHandModel(model_name(h, a.side))
        idx = m.data_index(names[h])                                # model dof d = LAD column idx[d]
        assert sorted(idx) == list(range(len(names[h]))), f"{h}: LAD joints and model DoF differ"
        pat = f"{TELEOP}/*/{h}_right/*_right/meta/info.json" if a.side == "right" else f"{TELEOP}/*/{h}_bimanual/*/meta/info.json"
        qs, tasks = [], []
        for inf in sorted(glob.glob(pat)):
            ds = os.path.dirname(os.path.dirname(inf))
            act, cols = load_actions(ds)
            q = np.zeros((len(act), len(idx)), np.float32)
            q[:, idx] = act[:, m.data_index(cols)]                  # teleop finger actions (rad) -> LAD column order
            qs.append(q); tasks.append(ds.split("/")[-3])
        q = np.concatenate(qs)
        lo, hi = m._lower.numpy(), m._upper.numpy()
        inside = float(((q[:, idx] >= lo - 1e-3) & (q[:, idx] <= hi + 1e-3)).mean())
        if len(q) > a.n_teleop:
            q = q[np.sort(rng.choice(len(q), a.n_teleop, replace=False))]
        out[f"teleop_{h}"] = q
        out[f"pair_{h}"] = P[f"{a.side}_{h}"].astype(np.float32)
        info[h] = {"teleop_frames": int(len(q)), "teleop_tasks": tasks, "teleop_frac_inside_limits": round(inside, 5),
                   "teleop_joint_std_min": round(float(q.std(0).min()), 4), "pair_rows": int(len(out[f"pair_{h}"]))}
        print(h, json.dumps(info[h]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/inputs.npz", hands=np.asarray(hands), names=json.dumps(names), side=a.side, info=json.dumps(info), **out)


def decode(a):
    from types import SimpleNamespace
    from dexverse_lad.train_cam import make_model
    I = np.load(f"{a.out}/inputs.npz", allow_pickle=True)
    hands = [str(h) for h in I["hands"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["hands"] == hands, (ck["hands"], hands)
        model, _ = make_model({h: I[f"pair_{h}"].shape[1] for h in hands}, SimpleNamespace(**ck["args"]))
        model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
        mean = {h: torch.tensor(ck["normalization"][h]["mean"], dtype=torch.float32, device=dev) for h in hands}
        std = {h: torch.tensor(ck["normalization"][h]["std"], dtype=torch.float32, device=dev) for h in hands}
        out = {}
        with torch.no_grad():
            for kind in ("teleop", "pair"):
                for s in hands:
                    x = torch.tensor(I[f"{kind}_{s}"], device=dev)
                    z = model.encoders[s]((x - mean[s]) / std[s])
                    for t in hands:                                 # every target hand (s == t = self reconstruction)
                        out[f"{kind}_{s}_to_{t}"] = (model.decoders[t](z) * std[t] + mean[t]).cpu().numpy()
        np.savez(f"{a.out}/decoded_{name}.npz", ckpt=path, **out)
        print("decoded", name, path, flush=True)


def decode_xlvla(a):
    """Same output format as decode(), for an XL-VLA hand latent checkpoint (dexlatent env): comparison on the same inputs."""
    from dexverse_xlvla.eval_latent import dec, enc, load_eval
    I = np.load(f"{a.out}/inputs.npz", allow_pickle=True)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        tr = load_eval(path)
        mn = {h: model_name(h, side) for h in hands}
        assert all(mn[h] in tr.hand_names for h in hands), (tr.hand_names, hands)
        M = {h: tr.hand_models[mn[h]] for h in hands}
        idx = {h: M[h].data_index(names[h]) for h in hands}
        out = {}
        with torch.no_grad():
            for kind in ("teleop", "pair"):
                for s in hands:
                    z = enc(tr, mn[s], M[s].angles_to_normalized(torch.tensor(I[f"{kind}_{s}"][:, idx[s]])).clamp(-1, 1))
                    for t in hands:
                        qn = dec(tr, mn[t], z).clamp(-1, 1)
                        ang = ((qn + 1) * 0.5 * (M[t]._upper - M[t]._lower) + M[t]._lower).numpy()      # model order, rad
                        q = np.zeros_like(ang); q[:, idx[t]] = ang                                         # -> LAD column order
                        out[f"{kind}_{s}_to_{t}"] = q
        np.savez(f"{a.out}/decoded_{name}.npz", ckpt=path, **out)
        print("decoded", name, path, flush=True)


def metrics(a):
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.eval_latent import line_err
    I = np.load(f"{a.out}/inputs.npz", allow_pickle=True)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
    M = {h: H.CanonicalHandModel(model_name(h, side)) for h in hands}
    idx = {h: M[h].data_index(names[h]) for h in hands}

    def tips(h, q):                                                 # LAD-order joints (rad) -> canonical fingertips (m)
        return M[h].forward(M[h].angles_to_normalized(torch.tensor(q[:, idx[h]])).clamp(-1, 1))

    summary = {}
    with torch.no_grad():
        ref = {k: {h: tips(h, I[f"{k}_{h}"]) for h in hands} for k in ("teleop", "pair")}
        for f in sorted(glob.glob(f"{a.out}/decoded_*.npz")):
            name = os.path.basename(f)[8:-4]; D = np.load(f, allow_pickle=True)
            out = {h: {} for h in hands}
            for s in hands:
                q = D[f"teleop_{s}_to_{s}"]                                                           # 1
                out[s]["m1_teleop_self_joint_deg"] = float(np.rad2deg(np.abs(q - I[f"teleop_{s}"])).mean())
                out[s]["m1_teleop_self_tip_mm"] = float((tips(s, q) - ref["teleop"][s]).norm(dim=-1).mean() * 1e3)
                j, tp, dd, ll = [], [], [], []
                for t in hands:
                    q = D[f"pair_{s}_to_{t}"]                                                         # 2
                    ej = float(np.rad2deg(np.abs(q - I[f"pair_{t}"])).mean())
                    et = float((tips(t, q) - ref["pair"][t]).norm(dim=-1).mean() * 1e3)
                    if s == t:
                        out[s]["pair_self_joint_deg"], out[s]["pair_self_tip_mm"] = ej, et
                        continue
                    j.append(ej); tp.append(et)
                    k = min(M[s].tip_count(), M[t].tip_count())                                       # 3
                    d, l = line_err(ref["teleop"][s], tips(t, D[f"teleop_{s}_to_{t}"]), [(0, i) for i in range(1, k)])
                    dd.append(d); ll.append(l * 1e3)
                out[s].update(m2_cross_joint_deg=float(np.mean(j)), m2_cross_tip_mm=float(np.mean(tp)),
                              m3_teleop_line_dir_deg=float(np.mean(dd)), m3_teleop_line_len_mm=float(np.mean(ll)))
            out["mean"] = {k: float(np.mean([out[h][k] for h in hands])) for k in out[hands[0]]}
            out["_ckpt"] = str(D["ckpt"]); out["_data"] = json.loads(str(I["info"]))
            json.dump(out, open(f"{a.out}/space_{name}.json", "w"), indent=1)
            summary[name] = out["mean"]
            print(name, json.dumps({k: round(v, 3) for k, v in out["mean"].items()}), flush=True)
    json.dump({"side": side, "mean": summary, "data": json.loads(str(I["info"]))}, open(f"{a.out}/space_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "decode", "decode_xlvla", "metrics"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_sub", default="")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n_teleop", type=int, default=30000)
    ap.add_argument("--ckpt", action="append", default=[])
    a = ap.parse_args()
    {"prep": prep, "decode": decode, "decode_xlvla": decode_xlvla, "metrics": metrics}[a.step](a)
