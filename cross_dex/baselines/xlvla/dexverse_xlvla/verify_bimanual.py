"""Checks before any bimanual pi0.5 training (CLAUDE.md "행동·상태 의미는 시뮬레이터에서 확인한 뒤 쓴다"), on one
(task, hand) bimanual teleop cell, with the simulator-read calibration of measure_wrist_bimanual.py:

  fk     : per side, the simulator's fingertip positions (fk_samples) vs calibration wrist pose x hand-model FK of the
           recorded finger joints -> tests root body, wrist kinematics, canonical frame and finger name mapping at once.
  check1 : per side, env action converted to joints (offset + scale * a) vs the recorded state at the same / next step
           (wrist as canonical pose: mm / deg; fingers: rad), and the pickle actions vs the LeRobot actions.
  check2 : (with --pkl of the same cell) canonical wrist poses of both hands at the end of every episode relative to the
           object's start position, and the distance between the two wrists; across hands with --compare (other
           calibrations / cells) -> the canonical coordinates agree across hands, raw wrist joints do not.
  check3 : per policy dim, std over the episode (0 / constant dims listed; padding dims excluded).
  replay : writes <out>/replay_{raw,joint,latent}/policy/server.json + actions.npz for ONE pickle episode: raw = recorded
           actions, joint = action -> joint policy vector -> action, latent = action -> [wrist | E(fingers)] ->
           [wrist | D(z)] -> action. Run them in the simulator with run_eval.py (common evaluator, video on).

  python -m dexverse_xlvla.verify_bimanual --task <t> --hand <h> --calib <wrist_calib_<t>.json> --ckpt <right latent>
         --ckpt_left <left latent> --pkl <bimanual pkl> --episode 0 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexverse_xlvla import bimanual as B
from dexverse_xlvla.make_dataset_bimanual import bimanual_dir


def _r(x, n=4):
    return float(np.round(float(x), n))


def fk_check(space: B.BimanualSpace, calib: dict) -> dict:
    out = {}
    for s in B.SIDES:
        c, m, d = calib["sides"][s], space.models[s], space.side[s]
        sim_tips = {}
        for t in m.tip_links:                                 # model tip link -> articulation body (name match)
            key = t.replace(".", "_")
            hit = [b for b in c["tip_bodies"] if b and (b == key or b.endswith("_" + key) or b == t)]
            sim_tips[t] = hit[0] if hit else None
        errs = []
        for smp in c["fk_samples"]:
            jp = smp["joint_pos"]
            q_w = np.array([jp[n] for n in c["action_joints"]])
            fj = np.array([jp[c["finger_joints"][i]] for i in range(len(c["finger_joints"]))])
            idx, _ = B.finger_index(m, c["finger_joints"], s)
            qn = m.angles_to_normalized(torch.tensor(fj[idx][None], dtype=torch.float32)).clamp(-1, 1)
            with torch.no_grad():
                tips_c = m.forward(qn)[0].numpy().astype(np.float64)        # canonical frame
            t_c = d["wrist"].pose(q_w)                                         # world pose of the canonical frame
            pred = tips_c @ t_c[:3, :3].T + t_c[:3, 3]
            for k, t in enumerate(m.tip_links):
                if sim_tips[t]:
                    errs.append((t, float(np.linalg.norm(pred[k] - np.asarray(smp["tips_world"][sim_tips[t]])))))
        e = np.array([x[1] for x in errs]) * 1000
        out[s] = {"model": m.hand_name, "finger_name_rule": d["rule"], "tips_matched": sim_tips,
                  "n": len(e), "tip_err_mm_mean": _r(e.mean(), 2) if len(e) else None,
                  "tip_err_mm_max": _r(e.max(), 2) if len(e) else None}
    return out


def pose_err(v1, v2):
    p = np.linalg.norm(v1[:, :3] - v2[:, :3], axis=1) * 1000
    from dexverse_xlvla.wrist import from_rot6d
    r1, r2 = from_rot6d(v1[:, 3:9]), from_rot6d(v2[:, 3:9])
    ang = np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", r1, r2)).magnitude())
    return p, ang


def check1(space, calib, act, st, pkl_act) -> dict:
    out = {"pkl_vs_lerobot_action_maxabs": _r(np.abs(pkl_act - act).max(), 8) if pkl_act is not None else None}
    for s in B.SIDES:
        d = space.side[s]
        W = d["wrist"]
        va = W.vec_from_actions(act[:, d["wdims"]])
        vs = W.vec_from_joints(st[:, d["wdims"]])
        qa = d["offset"] + d["scale"] * act[:, d["cols"]]
        qs = st[:, d["cols"]]
        r = {}
        for lag in (0, 1):
            n = len(act) - lag
            p, ang = pose_err(va[:n], vs[lag:lag + n])
            r[f"lag{lag}"] = {"wrist_pos_mm_median": _r(np.median(p), 3), "wrist_pos_mm_p95": _r(np.percentile(p, 95), 3),
                              "wrist_rot_deg_median": _r(np.median(ang), 3), "wrist_rot_deg_p95": _r(np.percentile(ang, 95), 3),
                              "finger_rad_median": _r(np.median(np.abs(qa[:n] - qs[lag:lag + n])), 4),
                              "finger_rad_p95": _r(np.percentile(np.abs(qa[:n] - qs[lag:lag + n]), 95), 4)}
        # the wrist drive follows its target with a delay: canonical-pose error over lags (a delay, not a semantic
        # mismatch, has a clear minimum at some lag > 0)
        scan = {}
        for lag in (0, 1, 2, 3, 5, 8, 12, 20, 30):
            n = len(act) - lag
            p, ang = pose_err(va[:n], vs[lag:lag + n])
            scan[lag] = (_r(np.median(p), 2), _r(np.median(ang), 2))
        r["wrist_lag_scan_median_mm_deg"] = scan
        # approach phase only (first quarter of the episode, before the hands hold the object): a target the hand cannot
        # reach because it pushes into the object shows up as a persistent offset later in the episode
        q4 = max(len(act) // 4, 13)
        p, ang = pose_err(va[:q4 - 12], vs[12:q4])
        r["wrist_lag12_first_quarter_median_mm_deg"] = (_r(np.median(p), 2), _r(np.median(ang), 2))
        p, ang = pose_err(va[-q4:-12], vs[-q4 + 12:])
        r["wrist_lag12_last_quarter_median_mm_deg"] = (_r(np.median(p), 2), _r(np.median(ang), 2))
        # raw joints per wrist dim (offset + scale * a vs state, lag 1) and the same without the offset
        ja = W.joints(act[:, d["wdims"]])
        r["wrist_joints"] = calib["sides"][s]["action_joints"]
        r["raw_wrist_joint_absdiff_median_lag1_per_dim"] = [_r(x, 4) for x in np.median(np.abs(ja[:-1] - st[1:, d["wdims"]]), 0)]
        r["raw_wrist_joint_absdiff_median_lag1_per_dim_if_no_offset"] = \
            [_r(x, 4) for x in np.median(np.abs(act[:-1, d["wdims"]] - st[1:, d["wdims"]]), 0)]
        r["finger_joint_std_rad"] = [_r(x, 4) for x in qs.std(0)]
        out[s] = r
    return out


def check3(vec, names, pad_mask) -> dict:
    sd = vec.std(0)
    flat = [n for n, v, p in zip(names, sd, pad_mask) if v < 1e-6 and not p]
    return {"std_min_nonpad": _r(sd[~pad_mask].min(), 6), "std_median_nonpad": _r(np.median(sd[~pad_mask]), 5),
            "constant_nonpad_dims": flat, "pad_dims": int(pad_mask.sum()),
            "absmax": _r(np.abs(vec).max(), 4)}


def object_start(ep) -> tuple[str, np.ndarray]:
    st = ep["initial_state"]
    cands = [(k, v["root_pose"]) for k, v in st.get("rigid_object", {}).items()
             if not any(w in k for w in ("table", "shelf", "leg", "rack"))]
    cands += [(k, v["root_pose"]) for k, v in st["articulation"].items() if k != "robot"]
    k, p = cands[0]
    return k, np.asarray(p, np.float64).reshape(-1)[:3]


def check2_cell(space, payload) -> dict:
    rel, raw, dist, dist_raw, name = {s: [] for s in B.SIDES}, {s: [] for s in B.SIDES}, [], [], None
    for ep in payload["episodes"]:
        a = np.asarray(ep["actions"], np.float64)
        name, p0 = object_start(ep)
        ends = {}
        for s in B.SIDES:
            d = space.side[s]
            v = d["wrist"].vec_from_actions(a[-1:, d["wdims"]])[0]
            ends[s] = v[:3]
            rel[s].append(v[:3] - p0)
            raw[s].append(d["wrist"].joints(a[-1, d["wdims"]])[:3])
        dist.append(np.linalg.norm(ends["right"] - ends["left"]))
        dist_raw.append(np.linalg.norm(raw["right"][-1] - raw["left"][-1]))
    f = lambda x: [_r(v, 4) for v in x]
    return {"object": name, "episodes": len(dist),
            **{f"{s}_canon_minus_object_mean_m": f(np.mean(rel[s], 0)) for s in B.SIDES},
            **{f"{s}_canon_minus_object_std_m": f(np.std(rel[s], 0)) for s in B.SIDES},
            **{f"{s}_raw_wrist_trans_joints_mean": f(np.mean(raw[s], 0)) for s in B.SIDES},
            "wrist_distance_canon_m_mean_std": f([np.mean(dist), np.std(dist)]),
            "wrist_distance_raw_joints_m_mean_std": f([np.mean(dist_raw), np.std(dist_raw)])}


def write_replay(out, name, actions, ep, max_steps):
    d = f"{out}/replay_{name}"
    os.makedirs(f"{d}/policy", exist_ok=True)
    np.savez(f"{d}/actions.npz", **{str(ep): actions.astype(np.float32)})
    json.dump({"mode": "replay", "actions": f"{d}/actions.npz", "episodes": [ep]}, open(f"{d}/policy/server.json", "w"))
    json.dump({"episode": ep, "max_steps": max_steps}, open(f"{d}/replay.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_left", required=True)
    ap.add_argument("--pkl", required=True, help="the pickle the LeRobot cell was converted from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    calib = B.load_calib(a.calib, a.hand)
    lat = B.build(a.calib, a.hand, "latent", a.ckpt, a.ckpt_left)
    jnt = B.BimanualSpace(calib, a.hand, lat.models, None)
    rep = {"task": a.task, "hand": a.hand, "calib": a.calib, "robot_type": calib["robot_type"],
           "action_joints": calib["action_joints"],
           "sides": {s: {k: calib["sides"][s][k] for k in ("wrist_dims", "finger_dims", "action_joints", "chain_joints",
                                                             "action_scale", "action_offset", "body",
                                                             "check_pos_err_max_m", "check_rot_err_max_deg")}
                     for s in B.SIDES}}
    rep["fk"] = fk_check(lat, calib)
    # dataset episode (check 1 / 3)
    import pandas as pd
    from dexverse_xlvla.make_dataset import names as dnames
    src = bimanual_dir(a.task, a.hand)
    info = json.load(open(f"{src}/meta/info.json"))
    rep["dataset"] = src
    from dexverse_xlvla.make_dataset_bimanual import names_ok
    rep["dataset_names_equal_sim_action_joints"] = names_ok(info, lat)
    rep["dataset_state_dims"] = len(dnames(info, "observation.state"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"])
                    for f in sorted(__import__("glob").glob(f"{src}/data/*/*.parquet"))])
    df = df[df["episode_index"] == a.episode].sort_values("frame_index")
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    payload = pickle.load(open(a.pkl, "rb"))
    pact = np.asarray(payload["episodes"][a.episode]["actions"], np.float64)
    rep["check1"] = check1(lat, calib, act, st, pact if pact.shape == act.shape else None)
    if pact.shape != act.shape:
        rep["check1"]["pkl_vs_lerobot_shape"] = [list(pact.shape), list(act.shape)]
    rep["check2"] = check2_cell(lat, payload)
    for name, sp in (("latent", lat), ("joint", jnt)):
        v = sp.to_policy(act, "action")
        pad = np.zeros(sp.dim, bool)
        if name == "joint":
            for i, s in enumerate(B.SIDES):
                o = i * (9 + sp.k) + 9
                pad[o + sp.models[s].dof_count(): o + sp.k] = True
        rep[f"check3_{name}"] = check3(v, sp.names(), pad)
    # replay actions from the PICKLE episode (the evaluator restores that episode's start)
    q0 = {s: lat.side[s]["wrist"].joints(pact[0, lat.side[s]["wdims"]]) for s in B.SIDES}
    conv = {"raw": pact}
    for name, sp in (("joint", jnt), ("latent", lat)):
        conv[name] = sp.to_action(sp.to_policy(pact, "action"), q0)
    rt = {}
    for name in ("joint", "latent"):
        e = {}
        for s in B.SIDES:
            d = lat.side[s]
            e[s] = {"wrist_action_maxabs": _r(np.abs(conv[name][:, d["wdims"]] - pact[:, d["wdims"]]).max(), 6),
                    "finger_action_rad_mean": _r(np.abs(conv[name][:, d["cols"]] - pact[:, d["cols"]]).mean(), 5),
                    "finger_action_rad_max": _r(np.abs(conv[name][:, d["cols"]] - pact[:, d["cols"]]).max(), 5)}
        rt[name] = e
    rep["roundtrip_action_error"] = rt
    ms = len(pact) + 120
    for name, x in conv.items():
        write_replay(a.out, name, x, a.episode, ms)
    rep["replay"] = {"episode": a.episode, "steps": len(pact), "max_steps": ms, "pkl": a.pkl,
                     "pkl_episode_success": bool(payload["episodes"][a.episode].get("success"))}
    json.dump(rep, open(f"{a.out}/verify.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("fk", "check1", "check2", "check3_latent", "check3_joint",
                                         "roundtrip_action_error", "dataset_names_equal_sim_action_joints")}, indent=1))


if __name__ == "__main__":
    main()
