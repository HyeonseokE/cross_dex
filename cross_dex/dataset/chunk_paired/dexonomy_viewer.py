"""Web viewer for the Dexonomy grasp dataset (Shadow hand, Feix GRASP taxonomy types), with the three key poses
pre-grasp -> grasp -> squeeze played as an interpolated trajectory around the object.

  /workspace/envs/dexverse/openpi/bin/python -m cross_dex.dataset.chunk_paired.dexonomy_viewer --root <dexonomy dir> --port 8771

<root> holds Dexonomy_GRASP_shadow/succ_collect/<feix#>_<name>/<obj>/floating/scaleXXX.npy (HF JiayiChenPKU/Dexonomy),
DGN_5k/{processed_data,scene_cfg} (object meshes) and hand/right.xml + hand/assets (Dexonomy shadow MJCF, menagerie meshes).
Hand FK = MuJoCo on that MJCF; the sample's qpos[:7] (pos, wxyz) is the pose of the hand MJCF world frame (Dexonomy
attaches its free joint there), qpos[7:] the 22 joints in MJCF order. Objects: scene_cfg mesh x scale x scene_scale[k].
Only objects of DGN_5k are listed (objaverse_5k meshes are not downloaded).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/datasets/dexverse/grasp_taxonomy/dexonomy")
ap.add_argument("--port", type=int, default=8771)
ap.add_argument("--per_type", type=int, default=40, help="objects listed per grasp type")
ap.add_argument("--fps", type=float, default=60.0)
args = ap.parse_args()

GRASP_DIR = os.path.join(args.root, "Dexonomy_GRASP_shadow", "succ_collect")
OBJ_ROOT = os.path.join(args.root, "DGN_5k")
MODEL = mujoco.MjModel.from_xml_path(os.path.join(args.root, "hand", "right.xml"))
DATA = mujoco.MjData(MODEL)
VIS = [g for g in range(MODEL.ngeom) if MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and MODEL.geom_group[g] == 2]
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dexonomy_viewer.html"), "rb").read()
_LIST: dict[str, list[str]] = {}
_LOCK = threading.Lock()   # one MjData shared by the server threads


def types() -> list[dict]:
    out = []
    for d in sorted(os.listdir(GRASP_DIR), key=lambda s: int(s.split("_")[0])):
        n, name = d.split("_", 1)
        out.append({"dir": d, "feix": int(n), "name": name.replace("_", " ")})
    return out


def samples(tdir: str) -> list[str]:
    if tdir not in _LIST:
        objs = [o for o in sorted(os.listdir(os.path.join(GRASP_DIR, tdir)))
                if os.path.isdir(os.path.join(OBJ_ROOT, "processed_data", o))][: args.per_type]
        _LIST[tdir] = [os.path.relpath(f, GRASP_DIR) for o in objs
                       for f in sorted(glob.glob(os.path.join(GRASP_DIR, tdir, o, "*", "*.npy")))]
    return _LIST[tdir]


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def trajectory(keys: np.ndarray, durs=(0.6, 0.3), hold=0.3) -> np.ndarray:
    """keys [3, 29] (pre-grasp, grasp, squeeze) -> [T, 29]: hold, min-jerk segments (joints, position; slerp root), hold."""
    out = [np.repeat(keys[:1], int(hold * args.fps), 0)]
    for a, b, d in zip(keys[:-1], keys[1:], durs):
        s = min_jerk(np.linspace(0, 1, max(int(d * args.fps), 2)))
        seg = a[None] + s[:, None] * (b - a)[None]
        rots = R.from_quat(np.stack([a[3:7], b[3:7]])[:, [1, 2, 3, 0]])
        q = Slerp([0, 1], rots)(s).as_quat()[:, [3, 0, 1, 2]]
        seg[:, 3:7] = q
        out.append(seg)
    out.append(np.repeat(keys[-1:], int(hold * args.fps), 0))
    return np.concatenate(out)


def hand_frames(traj: np.ndarray) -> list[list[list[float]]]:
    """Per frame, per visual geom: [x, y, z, qw, qx, qy, qz] in world."""
    frames = []
    with _LOCK:
        return _hand_frames(traj, frames)


def _hand_frames(traj, frames):
    for q in traj:
        DATA.qpos[:] = q[7:]
        mujoco.mj_kinematics(MODEL, DATA)
        Rr = R.from_quat(q[[4, 5, 6, 3]])
        row = []
        for g in VIS:
            p = Rr.apply(DATA.geom_xpos[g]) + q[:3]
            r = (Rr * R.from_matrix(DATA.geom_xmat[g].reshape(3, 3))).as_quat()
            row.append([round(float(v), 5) for v in (*p, r[3], r[0], r[1], r[2])])
        frames.append(row)
    return frames


def sample(rel: str, k: int) -> dict:
    d = np.load(os.path.join(GRASP_DIR, rel), allow_pickle=True).item()
    k = min(k, len(d["grasp_qpos"]) - 1)
    keys = np.stack([d["pregrasp_qpos"][k], d["grasp_qpos"][k], d["squeeze_qpos"][k]]).astype(np.float64)
    traj = trajectory(keys)
    scene_path = os.path.join(OBJ_ROOT, d["scene_path"].split("/DGN_5k/", 1)[1])
    sc = np.load(scene_path, allow_pickle=True).item()
    obj = next(iter(sc["scene"].values()))
    mesh = os.path.normpath(os.path.join(os.path.dirname(scene_path), obj["file_path"]))
    scale = np.asarray(obj["scale"], float) * (float(d["scene_scale"][k]) if "scene_scale" in d else 1.0)
    return {"rel": rel, "k": k, "n_grasps": int(len(d["grasp_qpos"])), "fps": args.fps,
            "key_frames": [int(0.3 * args.fps), int(0.3 * args.fps) + int(0.6 * args.fps) - 1, len(traj) - int(0.3 * args.fps) - 1],
            "geoms": [int(MODEL.geom_dataid[g]) for g in VIS], "frames": hand_frames(traj),
            "joints": np.round(traj[:, 7:], 4).tolist(),
            "object": {"mesh": os.path.relpath(mesh, OBJ_ROOT), "scale": scale.tolist(), "pose": np.asarray(obj["pose"], float).tolist()}}


def mesh_json(i: int) -> dict:
    va, vn = MODEL.mesh_vertadr[i], MODEL.mesh_vertnum[i]
    fa, fn = MODEL.mesh_faceadr[i], MODEL.mesh_facenum[i]
    return {"v": np.round(MODEL.mesh_vert[va: va + vn].ravel(), 5).tolist(), "f": MODEL.mesh_face[fa: fa + fn].ravel().tolist()}


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        p, q = unquote(u.path), {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if p in ("/", "/index.html"):
                return self._send(HTML, "text/html; charset=utf-8")
            if p == "/api/types":
                return self._send(types())
            if p == "/api/samples":
                return self._send(samples(q["type"]))
            if p == "/api/sample":
                return self._send(sample(q["rel"], int(q.get("k", 0))))
            if p.startswith("/api/mesh/"):
                return self._send(mesh_json(int(p.rsplit("/", 1)[1])))
            if p.startswith("/obj/"):
                f = os.path.normpath(os.path.join(OBJ_ROOT, p[len("/obj/"):]))
                if not f.startswith(OBJ_ROOT) or not os.path.isfile(f):
                    return self._send(b"missing", "text/plain", 404)
                return self._send(open(f, "rb").read(), "text/plain")
            return self._send(b"not found", "text/plain", 404)
        except Exception as e:  # noqa: BLE001
            return self._send(repr(e).encode(), "text/plain", 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {args.root} on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
