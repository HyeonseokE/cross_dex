"""Open-loop probe of a multi-hand latent DP on a hand's own demos (RSS2027 B-experiment diagnosis, 2026-11-23; dexlatent venv).

Feeds the recorded frames of <hand>'s converted set (images + [wrist 9 | E_h(q)] state, the DP's training format) to the
policy exactly as the evaluation server does (same Server class: pre/post processors, 240x320 resize, action queue), and
compares each executed policy output with the recorded target [wrist 9 | z*]:
  wrist_pos_mm   |pos - pos*| of the canonical wrist pose (first 3 values, metres -> mm)
  wrist_rot      |rot6 - rot6*| (L2 of the 6-D rotation part)
  z_l2           |z - z*|
  tip_mm         fingertip distance of D_h(z) vs D_h(z*) (policy error only, space error excluded)
--state_via <seen hand v> (input bypass, 2026-11-23): the finger part of the state is replaced by v's encoding of the same
pose, E_v(DexPilot(D_h(z_state) -> v)), so the state lies where the policy saw v's states; images unchanged. With the
bypass the remaining error is the images' (and the bypass retarget's) share.
No simulator: this separates "the policy does not output the right actions for an unseen hand" from closed-loop effects.

  PYTHONPATH=/workspace/codes/dexverse/DexLatent:/workspace/codes/cross_dex python -m cross_dex.ours.crosshand_dp.probe_open \
     --arm xlvla --hand shadow --policy <pretrained_model> --episodes 0-4 --json <out.json>
"""
import argparse
import json

import numpy as np
import torch

from cross_dex.ours.crosshand_dp.server import Server

TASK = "Dexverse-GraspKettle-v0"
WS = "/workspace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", default="0-4")
    ap.add_argument("--state_via", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import dec
    from dexverse_xlvla.replay import episode_list
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = f"{WS}/datasets/dexverse/crosshand_dp/{a.arm}/{a.hand}"
    cfg = {"arm": a.arm, "hand": a.hand, "task": TASK, "policy_ckpt": a.policy,
           "latent_ckpt": f"{WS}/checkpoints/dexverse/crosshand_dp/spaces/{a.arm}/model.pt",
           "wrist": f"{WS}/checkpoints/dexverse/xlvla/hf6/latent/wrist/wrist_calib_{TASK}.json", "log_dir": None}
    S = Server(cfg)
    keys = list(S.policy.config.input_features)
    eps = episode_list(a.episodes)
    ds = LeRobotDataset("local/probe", root=root)   # episodes= subset breaks on these sets (parquet column order)
    epcol = np.asarray(ds.hf_dataset["episode_index"])
    M = S.m
    rows = {k: [] for k in ("wrist_pos_mm", "wrist_rot", "z_l2", "tip_mm")}
    per_ep = {}
    for e in eps:
        S.reset()
        idx = np.nonzero(epcol == e)[0]
        er = {k: [] for k in rows}
        via = None
        if a.state_via:
            from dexverse_xlvla.eval_latent import enc
            from cross_dex.ours.uniform_pairs import retarget
            st = torch.stack([ds[int(i)]["observation.state"] for i in idx]).float()
            with torch.no_grad():
                q_obs = dec(S.tr, a.hand, st[:, 9:]).clamp(-1, 1)
            Mv = S.tr.hand_models[a.state_via]
            qv, _, _ = retarget(M, Mv, q_obs.cuda(), 300, 2, 0.05, 1e-6,
                                {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2})
            with torch.no_grad():
                via = enc(S.tr, a.state_via, qv.detach().cpu().clamp(-1, 1))
        for j, i in enumerate(idx):
            it = ds[int(i)]
            if via is not None:
                it["observation.state"] = torch.cat([it["observation.state"][:9], via[j]]).to(it["observation.state"].dtype)
            batch = {k: it[k].unsqueeze(0) if it[k].ndim == 3 else it[k] for k in keys}
            with torch.inference_mode():
                out = S.post(S.policy.select_action(S.pre(batch)))[0].cpu().float()
            tgt = it["action"].float()
            er["wrist_pos_mm"].append(float((out[:3] - tgt[:3]).norm()) * 1000)
            er["wrist_rot"].append(float((out[3:9] - tgt[3:9]).norm()))
            er["z_l2"].append(float((out[9:] - tgt[9:]).norm()))
            with torch.no_grad():
                qa = dec(S.tr, a.hand, out[None, 9:]).clamp(-1, 1); qb = dec(S.tr, a.hand, tgt[None, 9:]).clamp(-1, 1)
                er["tip_mm"].append(float((M.forward(qa) - M.forward(qb)).norm(dim=-1).mean()) * 1000)
        per_ep[e] = {k: float(np.mean(v)) for k, v in er.items()}
        for k in rows:
            rows[k] += er[k]
        print(e, {k: round(v, 3) for k, v in per_ep[e].items()}, flush=True)
    res = {"arm": a.arm, "hand": a.hand, "state_via": a.state_via or None, "policy": a.policy, "episodes": eps, "frames": len(rows["tip_mm"]),
           "mean": {k: float(np.mean(v)) for k, v in rows.items()},
           "p90": {k: float(np.percentile(v, 90)) for k, v in rows.items()}, "per_episode": per_ep}
    print(json.dumps(res["mean"]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
