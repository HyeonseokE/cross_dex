"""Open-loop replay of demonstrations through the chunk latent (check ④ before DP training; and the cross-hand latent replay
of XL-VLA Tab. 4 / LAD). Same outputs as dexverse_xlvla.replay (called for its helpers, not modified):

  actions.npz  per episode, hand B's action sequence: fingers = D_B(E_A(A's finger chunk starting at t)), chunks re-planned
               every --exec steps (first --exec steps of each decoded H-step chunk, as the DP executes);
               wrist = A's own wrist actions (A == B) or B's wrist actions reaching A's canonical wrist pose (A != B)
  demos.pkl    (A != B) B's pickle with A's scene per episode (dexverse_xlvla/build_demos.py, simulator env)
  policy/      server.json for dp_server.py (mode replay)

  /workspace/envs/dexverse/dexlatent/bin/python -m cross_dex.latent.chunk_latent.replay --task <T> --src shadow --dst wuji \
      --latent <best.pt> --xlvla_ckpt <XL-VLA .pt> --wrist <wrist_calib_<T>.json> --out <dir> --episodes 0-4
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess

import numpy as np

from cross_dex.latent.chunk_latent.model import HandCodec, load_model


def chunked(codec_a: HandCodec, codec_b: HandCodec, fa: np.ndarray, exec_steps: int) -> np.ndarray:
    """A finger targets [T, n_a] -> B finger targets [T, n_b] via chunk latents re-planned every exec_steps."""
    H, T = codec_a.H, len(fa)
    pad = np.concatenate([fa, np.repeat(fa[-1:], H, 0)])
    starts = list(range(0, T, exec_steps))
    z = codec_a.encode(np.stack([pad[t:t + H] for t in starts]))
    qb = codec_b.decode(z).cpu().numpy()                                  # [n_chunks, H, n_b]
    out = np.concatenate([qb[i, :exec_steps] for i in range(len(starts))])[:T]
    return out


def main():
    from dexverse_xlvla import hands as Hm
    from dexverse_xlvla.eval_latent import load
    from dexverse_xlvla.replay import MAIN_PYTHON, action_names, episode_list, pkl_path
    from dexverse_xlvla.wrist import load_wrists
    import dexverse_xlvla
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--xlvla_ckpt", required=True)
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--exec", type=int, default=8, help="executed steps per decoded chunk (DP n_action_steps)")
    ap.add_argument("--episodes", default="0-4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(f"{a.out}/policy", exist_ok=True)
    eps = episode_list(a.episodes)
    model, ck = load_model(a.latent, "cpu")
    ca, cb = HandCodec(model, ck, a.src), HandCodec(model, ck, a.dst)
    src_pkl = pkl_path(a.task, a.src)
    an, bn = action_names(a.task, a.src), action_names(a.task, a.dst)
    ia = [an.index(n) for n in ck["finger_joint_names"][a.src]]
    ib = [bn.index(n) for n in ck["finger_joint_names"][a.dst]]
    src = pickle.load(open(src_pkl, "rb"))
    cross = a.src != a.dst
    tr = load(a.xlvla_ckpt)
    wrists = load_wrists(a.wrist, {h: tr.hand_models[h] for h in {a.src, a.dst}}) if cross else None
    actions, err, firsts = {}, {}, {}
    for k in eps:
        act = np.asarray(src["episodes"][k]["actions"], dtype=np.float64)
        out = np.zeros((len(act), len(bn)))
        out[:, ib] = chunked(ca, cb, act[:, ia], a.exec)
        if cross:
            wb = wrists[a.dst]
            out[:, :6] = wb.actions_from_vec(wrists[a.src].vec_from_actions(act[:, :6]), q_init=wb.joints(np.zeros(6)))
            firsts[k] = wb.joints(out[0, :6])
        else:
            out[:, :6] = act[:, :6]
            err[k] = float(np.abs(out[:, ib] - act[:, ia]).mean() * 57.2958)          # check ①: round-trip error (deg)
        actions[k] = out.astype(np.float32)
    np.savez(f"{a.out}/actions.npz", **{str(k): v for k, v in actions.items()})
    demos = src_pkl
    if cross:   # B's pickle with A's scene; B's start = converted first step (fingers: decoded first frame)
        mb = tr.hand_models[a.dst]
        alias = {v: kk for kk, v in Hm.HANDS[a.dst].data_name.items()}
        start = {}
        for k in eps:
            j = {n: float(v) for n, v in zip(bn[:6], firsts[k])}
            j.update({alias.get(n, n): float(actions[k][0, i]) for n, i in zip(ck["finger_joint_names"][a.dst], ib)})
            start[str(k)] = j
        demos = f"{a.out}/demos.pkl"
        spec = {"src_pkl": src_pkl, "dst_pkl": pkl_path(a.task, a.dst), "out": demos, "src_hand": a.src, "mode": "latent",
                "episodes": eps, "start": start}
        json.dump(spec, open(f"{a.out}/demos_spec.json", "w"))
        subprocess.run([MAIN_PYTHON, os.path.join(os.path.dirname(dexverse_xlvla.__file__), "build_demos.py"),
                        f"{a.out}/demos_spec.json"], check=True, env={"PATH": os.environ.get("PATH", "")})
        _ = mb
    json.dump({"mode": "replay", "actions": f"{a.out}/actions.npz", "episodes": eps},
              open(f"{a.out}/policy/server.json", "w"))
    json.dump({"src": a.src, "dst": a.dst, "latent": a.latent, "exec": a.exec, "episodes": eps,
               "max_steps": int(max(len(v) for v in actions.values())) + 120, "src_pkl": src_pkl, "demos": demos,
               "roundtrip_mae_deg": err}, open(f"{a.out}/replay.json", "w"), indent=1)
    print(json.dumps({"demos": demos, "roundtrip_mae_deg": err}))


if __name__ == "__main__":
    main()
