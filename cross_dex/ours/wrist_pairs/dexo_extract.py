"""Sample Dexonomy Shadow grasps and extract what cross-hand contact matching needs (RSS2027 wrist-pair pilot, 2026-11-23).

Per sampled grasp ("grasp" keyframe, object scale = scene cfg scale x per-grasp scene_scale, object at the origin):
Shadow hand-object contacts from MuJoCo (position, finger 0..4 = thumb,index,middle,ring,little or 5 = palm, distal flag,
distance), Shadow palm pose, knuckle and fingertip positions (all in the object frame), the object's convex pieces.
  python dexo_extract.py --root <dexonomy dir> --per_type 10 --out pilot_grasps.pkl      (venv with mujoco >= 3.2)
"""
import argparse
import glob
import os
import pickle
import random

import mujoco
import numpy as np

from dexonomy_contact import build, scene_file

FINGER = {"rh_th": 0, "rh_ff": 1, "rh_mf": 2, "rh_rf": 3, "rh_lf": 4}
KNUCKLE = ["rh_thbase", "rh_ffknuckle", "rh_mfknuckle", "rh_rfknuckle", "rh_lfknuckle"]
DISTAL = ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"]
TIP_OFF = [0.0275, 0.026, 0.026, 0.026, 0.026]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per_type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1", help="i/n: objects i::n of every type, per_type/n grasps each")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pi, pn = (int(x) for x in a.part.split("/")); a.per_type = -(-a.per_type // pn)
    base = os.path.join(a.root, "Dexonomy_GRASP_shadow/succ_collect")
    hand_xml = os.path.join(a.root, "_repo/assets/hand/shadow/right.xml")
    out = []
    for t in sorted(os.listdir(base), key=lambda x: int(x.split("_")[0])):
        objs = sorted(os.listdir(os.path.join(base, t))); rng.shuffle(objs); n = 0
        objs = objs[pi::pn]
        for o in objs:
            if n >= a.per_type:
                break
            fs = sorted(glob.glob(os.path.join(base, t, o, "*", "*.npy")))
            if not fs:
                continue
            f = rng.choice(fs); d = np.load(f, allow_pickle=True).item()
            sf = scene_file(a.root, str(d["scene_path"]))
            if not os.path.exists(sf):
                continue
            cfg = np.load(sf, allow_pickle=True).item(); oc = cfg["scene"][cfg["task"]["obj_name"]]
            k = rng.randrange(len(d["grasp_qpos"])); scale = np.asarray(oc["scale"], float) * float(d["scene_scale"][k])
            xml = os.path.join(os.path.dirname(sf), oc["xml_path"])
            try:
                model, data = build(hand_xml, xml, np.asarray(oc["pose"], float), scale, 0.01)
            except Exception as e:   # noqa: BLE001
                print("skip", f, str(e)[:80]); continue
            q = np.asarray(d["grasp_qpos"][k], float); data.qpos[:len(q)] = q; mujoco.mj_forward(model, data)
            cons = []
            for c in data.contact[:data.ncon]:
                b1 = model.body(model.geom_bodyid[c.geom1]).name; b2 = model.body(model.geom_bodyid[c.geom2]).name
                if b1.startswith("hand-") == b2.startswith("hand-") or c.dist > a.tol:
                    continue
                hb = (b1 if b1.startswith("hand-") else b2)[5:]
                fi = next((v for p, v in FINGER.items() if hb.startswith(p)), 5)
                if hb == "rh_lfmetacarpal":
                    fi = 5
                cons.append((fi, hb.endswith("distal"), np.array(c.pos), float(c.dist), hb))
            bp = lambda nm: np.array(data.body("hand-" + nm).xpos)   # noqa: E731
            tips = np.stack([bp(b) + np.array(data.body("hand-" + b).xmat).reshape(3, 3)[:, 2] * off for b, off in zip(DISTAL, TIP_OFF)])
            T = np.eye(4); T[:3, :3] = np.array(data.body("hand-rh_palm").xmat).reshape(3, 3); T[:3, 3] = bp("rh_palm")
            out.append({"type": t, "file": os.path.relpath(f, a.root), "k": k, "scale": scale, "obj_pose": np.asarray(oc["pose"], float),
                        "pieces": os.path.join(os.path.dirname(xml), "meshes"), "palm_T": T, "knuckles": np.stack([bp(b) for b in KNUCKLE]),
                        "tips": tips, "contacts": cons, "shadow_q": q})
            n += 1
        print(t, n, flush=True)
    pickle.dump(out, open(a.out, "wb"))
    print("saved", len(out), "grasps; mean contacts", np.mean([len(g["contacts"]) for g in out]),
          "distal fingers", np.mean([len({c[0] for c in g["contacts"] if c[1]}) for g in out]))


if __name__ == "__main__":
    main()
