"""Contact check of Dexonomy Shadow grasps against their objects (RSS2027 wrist-pair data study, 2026-11-23).

The released grasp files hold only hand qpos (root pos 3 | quat wxyz 4 | 22 joints; pregrasp / grasp / squeeze) and the
scene path; contacts are not stored. This rebuilds the Dexonomy MuJoCo scene (hand MJCF of the Dexonomy repo with a free
joint + the object's CoACD convex pieces at its scene pose and scale, as dexonomy/sim/mujoco_env.py) and reads the
hand-object geom distances: per stage the number of hand bodies within --tol of the object, which bodies, the closest
distance and the deepest penetration. --scale_mode tests whether the per-grasp ``scene_scale`` multiplies the scene scale.

  python dexonomy_contact.py --root <dexonomy dir> --n 300 --json out.json      (env with mujoco; plain numpy otherwise)
"""
import argparse
import glob
import json
import os
import random

import mujoco
import numpy as np

STAGES = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")


def scene_file(root, scene_path):
    return os.path.join(root, scene_path.replace("assets/object/", "", 1))


def build(hand_xml, obj_xml, pose, scale, margin):
    spec = mujoco.MjSpec()
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
    hand = mujoco.MjSpec.from_file(hand_xml)
    for m in hand.meshes:
        m.file = os.path.join(os.path.dirname(hand_xml), hand.meshdir, m.file)
    hand.meshdir = ""
    for g in hand.geoms:
        g.margin = margin
    hw = spec.worldbody.add_frame().attach_body(hand.worldbody, "hand-", "")
    hw.add_freejoint(name="hand_freejoint")
    obj = mujoco.MjSpec.from_file(obj_xml)
    for m in obj.meshes:
        m.file = os.path.join(os.path.dirname(obj_xml), obj.meshdir, m.file)
        m.scale *= scale
    obj.meshdir = ""
    for g in obj.geoms:
        g.margin = margin
    ow = spec.worldbody.add_frame().attach_body(obj.worldbody, "obj-", "")
    ow.pos, ow.quat = pose[:3], pose[3:]
    model = spec.compile()
    return model, mujoco.MjData(model)


def contacts(model, data, q, tol):
    data.qpos[:len(q)] = q
    mujoco.mj_forward(model, data)
    per = {}
    for c in data.contact[:data.ncon]:
        b1 = model.body(model.geom_bodyid[c.geom1]).name; b2 = model.body(model.geom_bodyid[c.geom2]).name
        if b1.startswith("hand-") == b2.startswith("hand-"):
            continue
        hb = (b1 if b1.startswith("hand-") else b2)[5:]
        per[hb] = min(per.get(hb, 1.0), float(c.dist))
    near = {b: d for b, d in per.items() if d < tol}
    return {"n_bodies": len(near), "bodies": sorted(near), "min_dist_mm": 1000 * min(per.values()) if per else None,
            "tips": sum(("distal" in b or "tip" in b) for b in near)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hand_xml", default="")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--per_type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--scale_mode", default="cfg", choices=["cfg", "cfg_x_scene"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    hand_xml = a.hand_xml or os.path.join(a.root, "_repo/assets/hand/shadow/right.xml")
    rng = random.Random(a.seed)
    base = os.path.join(a.root, "Dexonomy_GRASP_shadow/succ_collect")
    files = []
    for t in sorted(os.listdir(base)):                       # a few random objects per grasp type (no full-tree glob)
        objs = sorted(os.listdir(os.path.join(base, t))); rng.shuffle(objs)
        for o in objs[:a.per_type]:
            fs = sorted(glob.glob(os.path.join(base, t, o, "*", "*.npy")))
            if fs:
                files.append(rng.choice(fs))
    rng.shuffle(files)
    rows = []
    for f in files:
        if len(rows) >= a.n:
            break
        d = np.load(f, allow_pickle=True).item()
        sf = scene_file(a.root, str(d["scene_path"]))
        if not os.path.exists(sf):
            continue
        cfg = np.load(sf, allow_pickle=True).item()
        o = cfg["scene"][cfg["task"]["obj_name"]]
        k = random.Random(a.seed + len(rows)).randrange(len(d["grasp_qpos"]))
        scale = np.asarray(o["scale"], float) * (float(d["scene_scale"][k]) if a.scale_mode == "cfg_x_scene" else 1.0)
        try:
            model, data = build(hand_xml, os.path.join(os.path.dirname(sf), o["xml_path"]), np.asarray(o["pose"], float), scale, a.margin)
        except Exception as e:   # noqa: BLE001
            print("skip", f, e); continue
        r = {"file": os.path.relpath(f, a.root), "k": k, "type": f.split("succ_collect/")[1].split("/")[0],
             "scene_scale": float(d["scene_scale"][k]), "root_dist_cm": 100 * float(np.linalg.norm(d["grasp_qpos"][k, :3]))}
        for s in STAGES:
            r[s[:-5]] = contacts(model, data, np.asarray(d[s][k], float), a.tol)
        rows.append(r)
        if len(rows) % 20 == 0:
            print("done", len(rows), flush=True)
    out = {"n": len(rows), "scale_mode": a.scale_mode, "tol_mm": a.tol * 1000}
    for s in ("pregrasp", "grasp", "squeeze"):
        nb = np.array([r[s]["n_bodies"] for r in rows]); md = np.array([r[s]["min_dist_mm"] if r[s]["min_dist_mm"] is not None else 99 for r in rows])
        tp = np.array([r[s]["tips"] for r in rows])
        out[s] = {"contact_rate": float((nb > 0).mean()), "bodies_mean": float(nb.mean()), "bodies_ge2": float((nb >= 2).mean()),
                  "tips_mean": float(tp.mean()), "min_dist_mm_p10_p50_p90": [float(x) for x in np.percentile(md, [10, 50, 90])],
                  "penetration_gt2mm": float((md < -2).mean())}
    cnt = {}
    for r in rows:
        for b in r["grasp"]["bodies"]:
            cnt[b] = cnt.get(b, 0) + 1
    out["grasp_body_freq"] = {b: round(c / len(rows), 2) for b, c in sorted(cnt.items(), key=lambda x: -x[1])}
    bt = {}
    for r in rows:
        bt.setdefault(r["type"], []).append(r["grasp"]["n_bodies"])
    out["grasp_bodies_by_type"] = {t: round(float(np.mean(v)), 1) for t, v in sorted(bt.items(), key=lambda x: int(x[0].split("_")[0]))}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump({"summary": out, "rows": rows}, open(a.json, "w"))


if __name__ == "__main__":
    main()
