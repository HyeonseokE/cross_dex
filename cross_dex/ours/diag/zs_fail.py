"""Where does a zero-shot rollout fail: wrist, fingers, or neither (RSS2027, 2026-11-24).

Input: the policy-server logs of an evaluation cell (run_cell.sh LOG_DIR: observed joints q per step) and the target hand's
own teleop demo of the SAME pickle episode (the evaluator restores that episode's initial state, so the kettle starts at
the same pose). Reference = the demo's hand at lift onset t_l (first step >= 15 the kettle is 1 cm above its height at step
10 for 5 steps). Both sides go through the same functions (wrist calib -> wrist pose of our canonical frame, hand model ->
fingertips in the canonical frame), so the numbers are differences of motion, not of conventions.
Per episode, over the rollout steps t:
  tip_world   min_t mean_f |tip_world(t) - tip_world_demo(t_l)|            (fingertips at the demo's grasp, t* = argmin)
  wrist       |wrist pos(t*) - demo|, rotation angle                        (wrist part of that error)
  shape       mean_f |tip_canonical(t*) - tip_canonical_demo(t_l)|         (finger part: same wrist frame, fingers only)
  wrist_min   min_t |wrist pos(t) - demo|                                   (did the wrist ever get there)
  lift        max_t z_wrist(t >= t*) - z_wrist(t*)                          (did the policy go on to lift)

  python zs_fail.py --cell <out dir with server.json, log/, eval/metrics.json> [--cell ...] --json out.json
"""
import argparse
import json
import pickle

import numpy as np
import torch


def load_cell(cell):
    from dexverse_xlvla.eval_latent import load
    from dexverse_xlvla.replay import action_names
    from dexverse_xlvla.wrist import from_rot6d, load_wrists
    cfg = json.load(open(f"{cell}/server.json")); met = json.load(open(f"{cell}/eval/metrics.json"))
    tr = load(cfg["latent_ckpt"]); h = cfg["hand"]; m = tr.hand_models[h]; names = action_names(cfg["task"], h); idx = m.data_index(names)
    wrist = load_wrists(cfg["wrist"], {h: m})[h]

    def kin(q):   # [T, >=n] joints in `names` order -> wrist pos [T,3], R [T,3,3], tips canonical [T,F,3], tips world [T,F,3]
        v = wrist.vec_from_joints(q[:, :6]); pos = np.asarray(v[:, :3], np.float64); R = np.stack([from_rot6d(np.asarray(x[3:9], np.float64)) for x in v])
        qn = m.angles_to_normalized(torch.tensor(q[:, :len(names)][:, idx], dtype=torch.float32)).clamp(-1, 1)
        with torch.no_grad():
            tc = m._forward_internal(qn)[0].double().numpy()
        return pos, R, tc, np.einsum("tij,tfj->tfi", R, tc) + pos[:, None]

    d = pickle.load(open(met["demos"], "rb")); pn = [n.replace(".", "_") for n in d["action_layout"]["robot_joint_names"]]
    col = [pn.index(n.replace(".", "_")) for n in names]
    rows = []
    for k, e in enumerate(met["episodes"]):
        S = d["episodes"][e["source_episode"]]["states"]
        jp = np.stack([np.asarray(s["articulation"]["robot"]["joint_position"])[0] for s in S]).astype(np.float64)[:, col]
        z = np.stack([np.asarray(s["rigid_object"]["object"]["root_pose"])[0] for s in S])[:, 2]
        up = z > z[min(10, len(z) - 1)] + 0.01; up[:15] = False; run = np.convolve(up, np.ones(5), "valid") == 5
        if not run.any():
            continue
        tl = int(np.argmax(run)); dp, dR, dtc, dtw = [x[0] for x in kin(jp[tl:tl + 1])]
        L = np.load(f"{cell}/log/ep{k:03d}.npz"); q = np.asarray(L["q"], np.float64); pos, R, tc, tw = kin(q)
        ew = np.linalg.norm(tw - dtw, axis=-1).mean(1); ts = int(np.argmin(ew)); wd = np.linalg.norm(pos - dp, axis=1)
        ang = np.degrees(np.arccos(np.clip((np.trace(R[ts].T @ dR) - 1) / 2, -1, 1)))
        dpos0 = np.linalg.norm(jp[0, :6] - q[0, :6])
        rows.append({"episode": e["source_episode"], "success": bool(e["success"]), "steps": int(e["steps"]), "demo_t_lift": tl, "t_star": ts,
                     "tip_world_mm": float(ew[ts] * 1e3), "wrist_mm": float(wd[ts] * 1e3), "wrist_deg": float(ang),
                     "shape_mm": float(np.linalg.norm(tc[ts] - dtc, axis=-1).mean() * 1e3), "wrist_min_mm": float(wd.min() * 1e3),
                     "lift_mm": float((pos[ts:, 2].max() - pos[ts, 2]) * 1e3), "demo_lift_mm": float((kin(jp[tl:])[0][:, 2].max() - dp[2]) * 1e3),
                     "start_wrist_joint_diff": float(dpos0),
                     "wrist_vec_mm": ((dR.T @ (pos[ts] - dp)) * 1e3).tolist()})   # wrist error in the demo's wrist frame
    return cfg, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell", action="append", required=True); ap.add_argument("--json", default="")
    a = ap.parse_args(); out = {}
    keys = ["tip_world_mm", "wrist_mm", "wrist_deg", "shape_mm", "wrist_min_mm", "lift_mm", "demo_lift_mm", "t_star", "demo_t_lift"]
    for c in a.cell:
        cfg, rows = load_cell(c); out[c] = rows
        print(f"\n== {c}\n   policy {cfg['policy_ckpt'].split('/')[-4]}  episodes {len(rows)}  success {sum(r['success'] for r in rows)}"
              f"  start wrist-joint diff max {max(r['start_wrist_joint_diff'] for r in rows):.4f}")
        for name, sel in (("success", True), ("fail", False)):
            R = [r for r in rows if r["success"] == sel]
            if R:
                print(f"   {name:8s} n={len(R):2d}  " + "  ".join(f"{k} {np.median([r[k] for r in R]):.1f}" for k in keys))
        V = np.array([r["wrist_vec_mm"] for r in rows if not r["success"]])
        if len(V):   # same direction in every episode (frame/convention offset) or scattered (policy)?
            print(f"   fail wrist error vector (demo wrist frame, mm): mean {np.round(V.mean(0), 1).tolist()} |mean| {np.linalg.norm(V.mean(0)):.1f}"
                  f"  mean|.| {np.linalg.norm(V, axis=1).mean():.1f}  std {np.round(V.std(0), 1).tolist()}")
        for r in rows:
            print(f"   ep {r['episode']:2d} {'S' if r['success'] else '-'} " + " ".join(f"{r[k]:7.1f}" for k in keys))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
