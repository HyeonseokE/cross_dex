"""Adopted latent-space metrics (experiments/a4/RSS2027/README.md "공유 잠재 공간 평가 메트릭") for LAD and XL-VLA models on
DexVerse and DexJoCo-X (the benchmark comes from the pair subsample: DexVerse teleop LeRobot sets, or the DexJoCo-X demos).
Metric 4 (cross-hand latent replay in the simulator) is not computed here.

  1  teleop self reconstruction   teleop finger actions -> encode -> decode on the same hand: joint error (deg), fingertip error (mm)
  2  cross reconstruction         held-out retargeting pair rows: latent of hand A decoded on hand B vs hand B's retargeted
                                  joints: joint error (deg), fingertip error (mm), mean over A != B
  3  cross geometric consistency  teleop frames: thumb-finger fingertip line of hand A vs of hand B decoded from A's latent:
                                  direction (deg), length (mm), mean over A != B (same definition as dexverse_xlvla.eval_latent.cross)
  5  random-joint self reconstruction   joints uniform in the URDF limits -> same hand: joint (deg), fingertip (mm)
  6  cross geometry on pinch / random   metric 3 on thumb-finger contact poses and on random joints: direction (deg), length (mm)
  7  latent continuity            decode(z + e) vs decode(z) on the same hand, demo frames, e ~ N(0, (0.05 * std of z per dim)^2)
                                  (noise relative to each space's own latent spread): joint (deg), fingertip (mm)
  8  latent alignment             held-out pair rows: hand A's latent finds hand B's latent of the same pose by cosine similarity
                                  among 4096 candidates (top-1 %, chance 0.024 %), mean over ordered pairs A != B and 4 blocks
  10 resolution                  demo pose q and q + d (d: random direction per frame, RMS 1 deg over the joints; the same d
                                  for every step of a chunk): does decode(enc(q + d)) - decode(enc(q)) reproduce d on the same
                                  hand? relative error |out change - d| / |d| (%) and fingertip-displacement error (mm)
  (pair_self_*: held-out pair rows, same hand = the reference column "retargeting pair self reconstruction")

Three steps, because the LAD model (robot_clip, omegaconf) and the hand FK models (DexLatent, urdf_parser_py) live in
different environments (no package is added to either):
  prep     (dexlatent env)  teleop frames of every hand in the LAD joint order + the pair rows        -> inputs.npz
  prep_extra (dexlatent env) random joints and pinch (thumb-finger contact) poses per hand            -> inputs_extra.npz
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


def model_name(h, side, bench="dexverse"):
    """Hand model (dexverse_xlvla.hands) of a pair-file hand: DexVerse URDFs, or the DexJoCo-X URDFs (dj_<hand>)."""
    return ("dj_" if bench == "dexjoco" else "") + h + ("" if side == "right" else "_left")


def bench_of(I):
    keys = I.files if hasattr(I, "files") else I
    return str(I["bench"]) if "bench" in keys else "dexverse"


def prep(a):
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.check_hands import load_actions
    P = np.load(a.pairs_sub, allow_pickle=True)
    names = json.loads(str(P["names"]))
    hands = [str(h) for h in P["hands"]]
    bench = str(P["bench"])
    rng = np.random.default_rng(1000)
    out, info = {}, {}
    for h in hands:
        m = H.CanonicalHandModel(model_name(h, a.side, bench))
        if a.side == "left" and bench == "dexverse":                # pair files carry the right-hand joint names
            try:
                m.data_index(names[h])
            except KeyError:
                names[h] = [n.replace("right_", "left_") for n in names[h]]
        idx = m.data_index(names[h])                                # model dof d = LAD column idx[d]
        assert sorted(idx) == list(range(len(names[h]))), f"{h}: LAD joints and model DoF differ"
        pat = f"{TELEOP}/*/{h}_right/*_right/meta/info.json" if a.side == "right" else f"{TELEOP}/*/{h}_bimanual/*/meta/info.json"
        qs, tasks = [], []
        if bench == "dexjoco":                                      # DexJoCo-X demos: actuator targets, pair files use the same order
            from dexverse_xlvla.eval_space_dexjoco import demo_frames
            dq, tasks = demo_frames(h, a.side, 10 ** 9, rng)
            q = np.zeros((len(dq), len(idx)), np.float32)
            q[:, idx] = dq[:, m.data_index(names[h])]
            qs.append(q)
        for inf in ([] if bench == "dexjoco" else sorted(glob.glob(pat))):
            ds = os.path.dirname(os.path.dirname(inf))
            act, cols = load_actions(ds)
            q = np.zeros((len(act), len(idx)), np.float32)
            q[:, idx] = act[:, m.data_index(cols)]                  # teleop finger actions (rad) -> LAD column order
            qs.append(q); tasks.append(ds.split("/")[-3])
        q = np.concatenate(qs) if qs else np.zeros((0, len(idx)), np.float32)      # no teleop set of this hand: metrics 1 / 3 skip it
        lo, hi = m._lower.numpy(), m._upper.numpy()
        inside = float(((q[:, idx] >= lo - 1e-3) & (q[:, idx] <= hi + 1e-3)).mean()) if len(q) else float("nan")
        if len(q) > a.n_teleop:
            q = q[np.sort(rng.choice(len(q), a.n_teleop, replace=False))]
        out[f"teleop_{h}"] = q
        out[f"pair_{h}"] = P[f"{a.side}_{h}"].astype(np.float32)
        info[h] = {"teleop_frames": int(len(q)), "teleop_tasks": tasks, "teleop_frac_inside_limits": round(inside, 5),
                   "teleop_joint_std_min": round(float(q.std(0).min()), 4) if len(q) else None, "pair_rows": int(len(out[f"pair_{h}"]))}
        print(h, json.dumps(info[h]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/inputs.npz", hands=np.asarray(hands), names=json.dumps(names), side=a.side, bench=bench, info=json.dumps(info), **out)


def prep_chunk(a):
    """Chunk inputs (DexVerse right hands of a chunk dataset, H steps) so frame spaces (XL-VLA, LAD) and chunk spaces (chunk VAE)
    are measured on the same data (frame spaces: every step on its own). demo = held-out teleop chunks (teleop_chunks.py npz),
    retargeting = the chunk dataset's validation split, random = linear interpolation between two random poses over the H
    steps, pinch = a thumb-finger contact pose held for the H steps."""
    import HandLatent.model as HM
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.train_latent import DexVerseTrainer
    sys_path_seed()
    from seed_control import seed_everything
    meta = json.load(open(f"{a.chunks}/meta.json")); hands = meta["hands"]; names = meta["finger_joint_names"]
    C = np.load(f"{a.chunks}/chunks.npz", mmap_mode="r"); Hs = int(meta["H"])
    rng = np.random.default_rng(1000)
    va = np.where(np.asarray(C["split"]) == 1)[0]; va = np.sort(rng.choice(va, min(a.n_pair_chunks, len(va)), replace=False))
    T = np.load(a.teleop_npz)
    H.install(); seed_everything(1000)
    tr = DexVerseTrainer(hands, HM.TrainingConfig(arm_dof=0, pinch_template_count=a.n_pinch, checkpoint_dir="/workspace/.tmp/space_eval_unused",
                                                  device=torch.device("cpu")))
    tr._cache_pinch_templates()
    g = torch.Generator().manual_seed(1000)
    out, extra, info = {}, {}, {}
    for h in hands:
        m = tr.hand_models[h]; idx = m.data_index(names[h])
        lo, hi = m._lower, m._upper
        ang = lambda qn: ((qn.clamp(-1, 1) + 1) * 0.5 * (hi - lo) + lo)
        q = T[f"q_{h}"]
        q = q[np.sort(rng.choice(len(q), min(a.n_demo_chunks, len(q)), replace=False))].astype(np.float32)
        out[f"teleop_{h}"] = q
        out[f"pair_{h}"] = np.asarray(C[f"q_{h}"][va], np.float32)
        a0, a1 = (torch.rand(a.n_random_chunks, m.dof_count(), generator=g) * 2 - 1 for _ in range(2))
        w = torch.linspace(0, 1, Hs).view(1, Hs, 1)
        r = ang(a0[:, None] * (1 - w) + a1[:, None] * w)                       # [N, H, dof] model order
        p = ang(tr._pinch_templates[h][:a.n_pinch_chunks])[:, None].expand(-1, Hs, -1)
        for k, v in (("random", r), ("pinch", p)):
            x = np.zeros((v.shape[0], Hs, len(idx)), np.float32); x[..., idx] = v.numpy(); extra[f"{k}_{h}"] = x
        info[h] = {"teleop_chunks": int(len(q)), "teleop_npz": a.teleop_npz, "pair_chunks": int(len(va)), "chunks": a.chunks,
                   "random_chunks": a.n_random_chunks, "pinch_chunks": a.n_pinch_chunks, "H": Hs}
        print(h, info[h], flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/inputs.npz", hands=np.asarray(hands), names=json.dumps(names), side="right", bench="dexverse", H=Hs,
             info=json.dumps(info), **out)
    np.savez(f"{a.out}/inputs_extra.npz", **extra)


def chunk_H(I):
    return int(I["H"]) if "H" in (I.files if hasattr(I, "files") else I) else 1


def decode_chunk(a):
    """Chunk VAE checkpoints (cross_dex.latent.chunk_latent) on chunk inputs, same output format as decode()."""
    from cross_dex.latent.chunk_latent.model import HandCodec, load_model
    I, kinds = load_inputs(a.out)
    hands = [str(h) for h in I["hands"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        model, ck = load_model(path, dev)
        assert set(hands) <= set(ck["hands"]), (ck["hands"], hands)
        C = {h: HandCodec(model, ck, h, dev) for h in hands}
        out = {}
        Hm, Hc = int(ck["H"]), chunk_H(I)                    # Hm = 1: frame variant of the chunk model (every step on its own)
        B = 8192                                              # batches (one call on all frames exceeds the CUDA launch limits)
        z_of = lambda s, kind: torch.cat([C[s].encode(x) for x in np.array_split(
            I[f"{kind}_{s}"].reshape(-1, Hm, I[f"{kind}_{s}"].shape[-1]), max(1, -(-len(I[f"{kind}_{s}"]) * Hc // Hm // B)))])
        dec_t = lambda t, z: torch.cat([C[t].decode(zb) for zb in z.split(B)]).cpu().numpy().reshape(-1, Hc, C[t].mean.shape[0])
        with torch.no_grad():
            for kind in kinds:
                for s in hands:
                    z = z_of(s, kind)
                    for t in hands:
                        out[f"{kind}_{s}_to_{t}"] = dec_t(t, z)
            extras(z_of, dec_t, hands, I, out, chunk=Hm > 1)
        np.savez(f"{a.out}/decoded_{name}.npz", ckpt=path, **out)
        print("decoded", name, path, flush=True)


def prep_extra(a):
    """Random joints (uniform in the URDF limits) and thumb-finger contact poses (DexVerseTrainer templates, seed 1000), per
    hand, in the LAD column order (rad). Separate from inputs.npz so the fixed demo / pair frames stay untouched."""
    import HandLatent.model as HM
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.train_latent import DexVerseTrainer
    sys_path_seed()
    I = np.load(f"{a.out}/inputs.npz", allow_pickle=True)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"]); bench = bench_of(I)
    H.install()
    mn = {h: model_name(h, side, bench) for h in hands}
    from seed_control import seed_everything
    seed_everything(1000)
    cfg = HM.TrainingConfig(arm_dof=0, pinch_template_count=a.n_pinch, checkpoint_dir="/workspace/.tmp/space_eval_unused",
                            device=torch.device("cpu"))
    tr = DexVerseTrainer([mn[h] for h in hands], cfg)
    tr._cache_pinch_templates()
    g = torch.Generator().manual_seed(1000)
    out = {}
    for h in hands:
        m = tr.hand_models[mn[h]]; idx = m.data_index(names[h])
        lo, hi = m._lower, m._upper
        ang = lambda qn: ((qn.clamp(-1, 1) + 1) * 0.5 * (hi - lo) + lo).numpy()
        for kind, qn in (("random", torch.rand(a.n_random, m.dof_count(), generator=g) * 2 - 1), ("pinch", tr._pinch_templates[mn[h]][:a.n_pinch])):
            q = np.zeros((len(qn), len(idx)), np.float32); q[:, idx] = ang(qn)
            out[f"{kind}_{h}"] = q
        print(h, {k: out[f"{k}_{h}"].shape for k in ("random", "pinch")}, "pinch stats", tr.template_stats[mn[h]], flush=True)
    np.savez(f"{a.out}/inputs_extra.npz", **out)


def sys_path_seed():
    import sys
    if "/workspace/experiments" not in sys.path:
        sys.path.insert(0, "/workspace/experiments")


RES_DEG = 1.0                                            # metric 10 perturbation size (RMS over the joints)


def unchunk(q, Hc):
    return q.reshape(-1, Hc, q.shape[-1]) if Hc > 1 else q


def flat(d):
    """Chunk arrays [N, H, n] -> frames [N * H, n] (metrics are per frame); other entries unchanged."""
    return {k: (v.reshape(-1, v.shape[-1]) if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in d.items()}


class Arrays(dict):
    """dict with the .files attribute of an npz archive."""
    @property
    def files(self):
        return list(self)


def load_inputs(out_dir):
    """inputs.npz (+ inputs_extra.npz when present) as one dict; kinds = the input sets available."""
    I = dict(np.load(f"{out_dir}/inputs.npz", allow_pickle=True))
    kinds = ["teleop", "pair"]
    g = np.random.default_rng(1000)                       # metric 10: perturbed demo poses (fixed, same for every space)
    for h in [str(x) for x in I["hands"]]:
        q = I[f"teleop_{h}"]
        if len(q):
            d = g.standard_normal((q.shape[0],) + (1,) * (q.ndim - 2) + (q.shape[-1],)).astype(np.float32)
            d *= np.deg2rad(RES_DEG) / np.sqrt((d ** 2).mean(-1, keepdims=True))
            I[f"teleopd_{h}"] = (q + d).astype(np.float32)
    if os.path.exists(f"{out_dir}/inputs_extra.npz"):
        I.update(np.load(f"{out_dir}/inputs_extra.npz", allow_pickle=True)); kinds += ["random", "pinch"]
    return I, kinds


def extras(z_of, dec_self, hands, I, out, chunk=False):
    """Metric 7 (decodes of noisy demo latents) and metric 8 inputs (pair latents: one row per pair row, or per chunk with
    chunk inputs = the chunk's latent flattened; frame spaces: the H frame latents concatenated). z_of(s, kind) -> latent
    tensor, dec_self(s, z) -> joints (rad, LAD column order, numpy)."""
    g = torch.Generator().manual_seed(1000)
    Hc = chunk_H(I)
    for s in hands:
        if len(I[f"teleop_{s}"]):
            z = z_of(s, "teleop")
            e = torch.randn(z.shape, generator=g).to(z.device) * 0.05 * z.std(0, keepdim=True)
            out[f"contbase_{s}"] = dec_self(s, z); out[f"cont_{s}"] = dec_self(s, z + e)
            if f"teleopd_{s}" in I:
                out[f"res_{s}"] = dec_self(s, z_of(s, "teleopd"))
        z = z_of(s, "pair").float()
        z = z.flatten(1) if chunk else z.reshape(-1, Hc * z.shape[-1])          # [chunks, K*d] or [chunks, H*d]
        out[f"zpair_{s}"] = z[:16384].cpu().numpy()


def decode(a):
    from types import SimpleNamespace
    from dexverse_lad.train_cam import make_model
    I, kinds = load_inputs(a.out)
    hands = [str(h) for h in I["hands"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert set(hands) <= set(ck["hands"]), (ck["hands"], hands)
        model, _ = make_model({h: len(ck["normalization"][h]["mean"]) for h in ck["hands"]}, SimpleNamespace(**ck["args"]))
        model.load_state_dict(ck["state_dict"]); model.to(dev).eval()
        mean = {h: torch.tensor(ck["normalization"][h]["mean"], dtype=torch.float32, device=dev) for h in hands}
        std = {h: torch.tensor(ck["normalization"][h]["std"], dtype=torch.float32, device=dev) for h in hands}
        out = {}
        Hc = chunk_H(I)
        z_of = lambda s, kind: model.encoders[s]((torch.tensor(I[f"{kind}_{s}"].reshape(-1, I[f"{kind}_{s}"].shape[-1]), device=dev) - mean[s]) / std[s])
        dec_t = lambda t, z: unchunk((model.decoders[t](z) * std[t] + mean[t]).cpu().numpy(), Hc)
        with torch.no_grad():
            for kind in kinds:
                for s in hands:
                    if not len(I[f"{kind}_{s}"]):
                        continue
                    z = z_of(s, kind)
                    for t in hands:                                 # every target hand (s == t = self reconstruction)
                        out[f"{kind}_{s}_to_{t}"] = dec_t(t, z)
            extras(z_of, dec_t, hands, I, out)
        np.savez(f"{a.out}/decoded_{name}.npz", ckpt=path, **out)
        print("decoded", name, path, flush=True)


def decode_xlvla(a):
    """Same output format as decode(), for an XL-VLA hand latent checkpoint (dexlatent env): comparison on the same inputs."""
    from dexverse_xlvla.eval_latent import dec, enc, load_eval
    I, kinds = load_inputs(a.out)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        tr = load_eval(path)
        mn = {h: model_name(h, side, bench_of(I)) for h in hands}
        assert all(mn[h] in tr.hand_names for h in hands), (tr.hand_names, hands)
        M = {h: tr.hand_models[mn[h]] for h in hands}
        idx = {h: M[h].data_index(names[h]) for h in hands}
        out = {}
        Hc = chunk_H(I)
        z_of = lambda s, kind: enc(tr, mn[s], M[s].angles_to_normalized(torch.tensor(I[f"{kind}_{s}"].reshape(-1, I[f"{kind}_{s}"].shape[-1])[:, idx[s]])).clamp(-1, 1))

        def dec_t(t, z):
            qn = dec(tr, mn[t], z).clamp(-1, 1)
            ang = ((qn + 1) * 0.5 * (M[t]._upper - M[t]._lower) + M[t]._lower).numpy()      # model order, rad
            q = np.zeros_like(ang); q[:, idx[t]] = ang                                         # -> LAD column order
            return unchunk(q, Hc)
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


def metrics(a):
    from dexverse_xlvla import hands as H
    from dexverse_xlvla.eval_latent import line_err
    I, kinds = load_inputs(a.out)
    I = flat(I)
    hands = [str(h) for h in I["hands"]]; names = json.loads(str(I["names"])); side = str(I["side"])
    M = {h: H.CanonicalHandModel(model_name(h, side, bench_of(I))) for h in hands}
    idx = {h: M[h].data_index(names[h]) for h in hands}

    def tips(h, q):                                                 # LAD-order joints (rad) -> canonical fingertips (m)
        return M[h].forward(M[h].angles_to_normalized(torch.tensor(q[:, idx[h]])).clamp(-1, 1))

    summary = {}
    with torch.no_grad():
        ref = {k: {h: tips(h, I[f"{k}_{h}"]) for h in hands} for k in kinds}
        for f in sorted(glob.glob(f"{a.out}/decoded_*.npz")):
            name = os.path.basename(f)[8:-4]; D0 = np.load(f, allow_pickle=True)
            D = Arrays(flat({k: D0[k] for k in D0.files}))
            out = {h: {} for h in hands}
            for s in hands:
                if f"teleop_{s}_to_{s}" in D.files:                                                   # 1 (no demo frames: skipped)
                    q = D[f"teleop_{s}_to_{s}"]
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
                    if not len(I[f"teleop_{s}"]):
                        continue
                    k = min(M[s].tip_count(), M[t].tip_count())                                       # 3
                    d, l = line_err(ref["teleop"][s], tips(t, D[f"teleop_{s}_to_{t}"]), [(0, i) for i in range(1, k)])
                    dd.append(d); ll.append(l * 1e3)
                out[s].update(m2_cross_joint_deg=float(np.mean(j)), m2_cross_tip_mm=float(np.mean(tp)),
                              m3_teleop_line_dir_deg=float(np.mean(dd)) if dd else float("nan"),
                              m3_teleop_line_len_mm=float(np.mean(ll)) if ll else float("nan"))
                if f"random_{s}_to_{s}" in D.files:                                                   # 5, 6
                    q = D[f"random_{s}_to_{s}"]
                    out[s]["m5_random_self_joint_deg"] = float(np.rad2deg(np.abs(q - I[f"random_{s}"])).mean())
                    out[s]["m5_random_self_tip_mm"] = float((tips(s, q) - ref["random"][s]).norm(dim=-1).mean() * 1e3)
                    for kind in ("pinch", "random"):
                        dd, ll = [], []
                        for t in hands:
                            if t == s:
                                continue
                            k = min(M[s].tip_count(), M[t].tip_count())
                            d, l = line_err(ref[kind][s], tips(t, D[f"{kind}_{s}_to_{t}"]), [(0, i) for i in range(1, k)])
                            dd.append(d); ll.append(l * 1e3)
                        out[s][f"m6_{kind}_line_dir_deg"] = float(np.mean(dd)); out[s][f"m6_{kind}_line_len_mm"] = float(np.mean(ll))
                if f"cont_{s}" in D.files:                                                            # 7
                    b, c = D[f"contbase_{s}"], D[f"cont_{s}"]
                    out[s]["m7_cont_joint_deg"] = float(np.rad2deg(np.abs(c - b)).mean())
                    out[s]["m7_cont_tip_mm"] = float((tips(s, c) - tips(s, b)).norm(dim=-1).mean() * 1e3)
                elif f"zpair_{s}" in D.files:
                    out[s]["m7_cont_joint_deg"] = out[s]["m7_cont_tip_mm"] = float("nan")
                if f"res_{s}" in D.files:                                                             # 10
                    din = I[f"teleopd_{s}"] - I[f"teleop_{s}"]; dout = D[f"res_{s}"] - D[f"contbase_{s}"]
                    out[s]["m10_res_rel_pct"] = float(100 * (np.linalg.norm(dout - din, axis=-1) / np.linalg.norm(din, axis=-1)).mean())
                    tin = tips(s, I[f"teleopd_{s}"]) - ref["teleop"][s]; tout = tips(s, D[f"res_{s}"]) - tips(s, D[f"contbase_{s}"])
                    out[s]["m10_res_tip_mm"] = float((tout - tin).norm(dim=-1).mean() * 1e3)
                if f"zpair_{s}" in D.files:                                                           # 8
                    acc = []
                    for t in hands:
                        if t == s:
                            continue
                        for b0 in range(0, len(D[f"zpair_{s}"]) - 4095, 4096):
                            zs = torch.nn.functional.normalize(torch.tensor(D[f"zpair_{s}"][b0:b0 + 4096]), dim=-1)
                            zt = torch.nn.functional.normalize(torch.tensor(D[f"zpair_{t}"][b0:b0 + 4096]), dim=-1)
                            acc.append(float(((zs @ zt.T).argmax(1) == torch.arange(len(zs))).float().mean()))
                    out[s]["m8_align_top1_pct"] = 100 * float(np.mean(acc))
            keys = list(dict.fromkeys(k for h in hands for k in out[h]))
            out["mean"] = {k: float(np.nanmean([out[h].get(k, float("nan")) for h in hands])) for k in keys}      # hands without teleop: nan, skipped
            out["_ckpt"] = str(D["ckpt"]); out["_data"] = json.loads(str(I["info"]))
            json.dump(out, open(f"{a.out}/space_{name}.json", "w"), indent=1)
            summary[name] = out["mean"]
            print(name, json.dumps({k: round(v, 3) for k, v in out["mean"].items()}), flush=True)
    json.dump({"side": side, "mean": summary, "data": json.loads(str(I["info"]))}, open(f"{a.out}/space_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "prep_extra", "prep_chunk", "decode", "decode_xlvla", "decode_chunk", "metrics"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_sub", default="")
    ap.add_argument("--side", default="right")
    ap.add_argument("--n_teleop", type=int, default=30000)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--n_random", type=int, default=4096)
    ap.add_argument("--n_pinch", type=int, default=1024)
    ap.add_argument("--chunks", default="", help="prep_chunk: chunk dataset dir (meta.json, chunks.npz with split)")
    ap.add_argument("--teleop_npz", default="", help="prep_chunk: held-out teleop chunks (teleop_chunks.py)")
    ap.add_argument("--n_demo_chunks", type=int, default=2000)
    ap.add_argument("--n_pair_chunks", type=int, default=8192)
    ap.add_argument("--n_random_chunks", type=int, default=512)
    ap.add_argument("--n_pinch_chunks", type=int, default=256)
    a = ap.parse_args()
    {"prep": prep, "prep_extra": prep_extra, "prep_chunk": prep_chunk, "decode": decode, "decode_xlvla": decode_xlvla,
     "decode_chunk": decode_chunk, "metrics": metrics}[a.step](a)
