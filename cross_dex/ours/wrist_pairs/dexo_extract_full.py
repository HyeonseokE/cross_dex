"""All Dexonomy Shadow grasps -> compact contact-matching inputs (RSS2027 wrist-pair data, 2026-11-23).

Like dexo_extract.py (MuJoCo contacts of the "grasp" keyframe, object scale = scene cfg scale x per-grasp scene_scale) but for
every grasp of every file, written as npz shards in the input format of contact_match_grid.py. The scene has to be rebuilt
per grasp because scene_scale differs per grasp. Resumable: (type, object) folders already in a shard of this part are skipped.
  python dexo_extract_full.py --root <dexonomy dir> --part i/n --out <dir>      (venv with mujoco >= 3.2)
"""
import argparse
import glob
import os

import mujoco
import numpy as np

from contact_match_grid_pack import contacts_arrays
from dexo_extract import DISTAL, FINGER, KNUCKLE, TIP_OFF
from dexonomy_contact import build, scene_file

KEYS = ("type", "src", "k", "scale", "obj", "palm_T", "shadow_q", "knuckles", "tips", "pc", "pm", "sc", "sm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--part", default="0/1"); ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.002); ap.add_argument("--shard", type=int, default=20000)
    ap.add_argument("--types", default="")
    a = ap.parse_args()
    pi, pn = (int(x) for x in a.part.split("/")); os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.root, "Dexonomy_GRASP_shadow/succ_collect"); hand_xml = os.path.join(a.root, "_repo/assets/hand/shadow/right.xml")
    old = sorted(glob.glob(f"{a.out}/x_{pi:03d}_*.npz")); done = set(); seq = len(old)
    for f in old:
        d = np.load(f); done |= {"/".join(x.decode().split("/")[2:4]) for x in np.unique(d["src"])}
    buf = {k: [] for k in KEYS}

    def flush():
        nonlocal seq
        if buf["k"]:
            tmp = f"{a.out}/x_{pi:03d}_{seq:04d}.tmp.npz"
            np.savez(tmp, **{k: (np.array(v, dtype="S") if k in ("type", "src", "obj") else np.asarray(v, np.int32 if k == "k" else np.float32))
                             for k, v in buf.items()})
            os.replace(tmp, tmp.replace(".tmp.npz", ".npz")); seq += 1
            for v in buf.values():
                v.clear()
    types = sorted(os.listdir(base), key=lambda x: int(x.split("_")[0]))
    for t in [x for x in types if not a.types or x.split("_")[0] in a.types.split(",")]:
        for o in sorted(os.listdir(os.path.join(base, t)))[pi::pn]:
            if f"{t}/{o}" in done:
                continue
            for f in sorted(glob.glob(os.path.join(base, t, o, "*", "*.npy"))):
                try:
                    d = np.load(f, allow_pickle=True).item(); sf = scene_file(a.root, str(d["scene_path"]))
                    cfg = np.load(sf, allow_pickle=True).item(); oc = cfg["scene"][cfg["task"]["obj_name"]]
                    xml = os.path.join(os.path.dirname(sf), oc["xml_path"])
                except Exception as e:   # noqa: BLE001
                    print("skip", f, str(e)[:80], flush=True); continue
                for k in range(len(d["grasp_qpos"])):
                    scale = np.asarray(oc["scale"], float) * float(d["scene_scale"][k])
                    try:
                        model, data = build(hand_xml, xml, np.asarray(oc["pose"], float), scale, 0.01)
                    except Exception as e:   # noqa: BLE001
                        print("skip", f, k, str(e)[:80], flush=True); continue
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
                    pc, pm, sc, sm = contacts_arrays(cons)
                    for key, v in zip(KEYS, (t, os.path.relpath(f, a.root), k, scale[0], os.path.normpath(os.path.join(os.path.dirname(xml), "meshes")),
                                             T, q, np.stack([bp(b) for b in KNUCKLE]), tips, pc, pm, sc, sm)):
                        buf[key].append(v)
            if len(buf["k"]) >= a.shard:
                flush()
        print(t, "done", flush=True)
    flush(); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
