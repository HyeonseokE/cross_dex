"""Failure taxonomy of cross-hand retargeted replays (RSS2027 step 0 of "beating the retargeting ceiling", 2026-11-23).

Reads the per-step logs of cross_pkl_replay.sh DIAG=1 (diag_replay.py: real success flag, object poses, joint q vs target,
fingertip-object contact forces) and classifies every episode:
  success     real task success at some step
  no_contact  no fingertip ever touches the object (> FORCE_N for >= 5 steps)     -> reach / wrist / finger shape
  no_lift     contact but the object never rises LIFT_M above its start           -> weak / wrong grasp
  dropped     lifted but ended below LIFT_M                                         -> slip in transport
  lifted_fail lifted and held at the end but no success (goal not reached)        -> transport / placement
plus: number of fingertips in contact at the moment of max contact, max lift, finger tracking error |q - q_target| (rad)
while in contact (fingers blocked by the object vs free), object horizontal push before lift.

  python rtdiag_analyze.py <out dir of cross_pkl_replay.sh> [...]   (plain python3 + numpy)
"""
import glob
import json
import os
import sys

import numpy as np

FORCE_N, LIFT_M = 0.5, 0.03


def episode(f):
    d = np.load(f, allow_pickle=True)
    obj = d["obj_object"]; z0 = obj[0, 2]; lift = obj[:, 2] - z0
    fk = sorted(k for k in d.files if k.startswith("force_"))
    F = np.concatenate([d[k] for k in fk], 1) if fk else np.zeros((len(obj), 0))
    touch = F > FORCE_N; n_touch = touch.sum(1)
    contact = n_touch > 0
    succ = bool(d["real"].any())
    lifted = lift.max() > LIFT_M
    if succ:
        kind = "success"
    elif contact.sum() < 5:
        kind = "no_contact"
    elif not lifted:
        kind = "no_lift"
    elif lift[-1] < LIFT_M:
        kind = "dropped"
    else:
        kind = "lifted_fail"
    t_lift = int(np.argmax(lift > LIFT_M)) if lifted else len(lift)
    push = float(np.linalg.norm(obj[:t_lift, :2] - obj[0, :2], axis=1).max()) if t_lift > 0 else 0.0
    q, qt = d["q"], d["q_target"]; fin = slice(6, q.shape[1])
    err = np.abs(q[:, fin] - qt[:, fin]).mean(1)
    return {"kind": kind, "success": succ, "max_lift_cm": float(lift.max() * 100), "final_lift_cm": float(lift[-1] * 100),
            "contact_steps": int(contact.sum()), "max_tips_in_contact": int(n_touch.max()) if len(n_touch) else 0,
            "tips": len(fk), "first_contact": int(np.argmax(contact)) if contact.any() else None,
            "push_before_lift_cm": push * 100, "track_err_contact": float(err[contact].mean()) if contact.any() else None,
            "track_err_free": float(err[~contact].mean()) if (~contact).any() else None, "steps": len(obj)}


def main():
    rows = {}
    for o in sys.argv[1:]:
        eps = {os.path.basename(f)[2:-4]: episode(f) for f in sorted(glob.glob(f"{o}/diag/ep*.npz"))}
        kinds = {}
        for e in eps.values():
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        rows[os.path.basename(o.rstrip("/"))] = {"episodes": eps, "kinds": kinds}
        print(os.path.basename(o.rstrip("/")), kinds)
        for k, e in eps.items():
            print(f"  ep{k} {e['kind']:12s} lift max {e['max_lift_cm']:5.1f} end {e['final_lift_cm']:5.1f} cm  tips {e['max_tips_in_contact']}/{e['tips']}"
                  f"  contact {e['contact_steps']:4d}  push {e['push_before_lift_cm']:4.1f} cm  track {e['track_err_contact']}")
    json.dump(rows, open("rtdiag_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
