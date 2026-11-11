"""Web viewer for retargeted DexCanvas episodes: every hand of the same episode side by side, robot mesh (retarget URDF
at the solved dex_q = the teleop retargeter's own finger solution) + the human hand in the retargeter's canonical frame
(x scaling, wrist on the robot palm link), synced playback.

  /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.viewer_server --rt <retarget out dir> --port 8770

Needs a retarget run that saved ``<hand>_dexq`` and ``urdf/<hand>.urdf`` (retarget_dexcanvas.py). Visual meshes are served
from the original URDF folder; meshes missing there are found by file name under ``--mesh_roots``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rt", required=True)
ap.add_argument("--port", type=int, default=8770)
ap.add_argument("--mesh_roots", default="/workspace/assets/dexverse,/workspace/assets/bench2dex",
                help="comma list searched by mesh file name when a mesh is missing next to the URDF")
args = ap.parse_args()

META = json.load(open(os.path.join(args.rt, "meta.json")))
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html"), "rb").read()
_MESH_INDEX: dict[str, list[str]] = {}
_INDEX_LOCK = threading.Lock()


def mesh_index() -> dict[str, list[str]]:
    # built once, under a lock: the browser requests many meshes in parallel, and a thread that saw a half-built index
    # answered 404 (shadow meshes vanished on a fresh server)
    with _INDEX_LOCK:
        if not _MESH_INDEX:
            idx: dict[str, list[str]] = {}
            for root in args.mesh_roots.split(","):
                for dp, _, fs in os.walk(root):
                    for f in fs:
                        if f.lower().endswith((".stl", ".obj", ".dae", ".glb", ".gltf", ".mtl")):
                            idx.setdefault(f, []).append(os.path.join(dp, f))
            _MESH_INDEX.update(idx)
    return _MESH_INDEX


def resolve_mesh(hand: str, rel: str) -> str | None:
    udir = os.path.dirname(META["hands"][hand]["urdf_orig"])
    cands = [os.path.join(udir, rel)] + [os.path.join(d, rel) for d in glob.glob(os.path.join(udir, "*/"))]
    for c in cands:
        if os.path.isfile(c):
            return c
    hits = mesh_index().get(os.path.basename(rel), [])
    hand_hits = [h for h in hits if hand in h.lower()]
    return (hand_hits or hits or [None])[0]


def episodes() -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(args.rt, "*.npz"))):
        with np.load(f) as z:
            if not all(f"{h}_dexq" in z.files for h in META["hands"]):
                continue
            out.append({"id": os.path.basename(f)[:-4], "label": str(z["action_label"]), "T": int(len(z["t"])),
                        "err_cm": {h: round(float(z[f"{h}_err"].mean()) * 100, 2) for h in META["hands"]}})
    return out


def episode(eid: str) -> dict:
    z = np.load(os.path.join(args.rt, f"{os.path.basename(eid)}.npz"))
    hands = {}
    for h, m in META["hands"].items():
        hands[h] = {"dex_joint_names": m["dex_joint_names"], "dexq": np.round(z[f"{h}_dexq"], 4).tolist(),
                    "err": np.round(z[f"{h}_err"].mean(1) * 100, 2).tolist(), "scaling": m["scaling"],
                    "palm_link": m["origin_links"][-1], "tip_links": m["task_links"][-5:],
                    "human": np.round(z[f"{h}_human_canonical"], 5).tolist() if f"{h}_human_canonical" in z.files else None,
                    "finger_k": m.get("finger_k")}
    return {"fps": META["fps"], "label": str(z["action_label"]), "human": np.round(z["human_canonical"], 5).tolist(),
            "hands": hands}


class H(SimpleHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(self.path.split("?")[0])
        try:
            if p in ("/", "/index.html"):
                return self._send(HTML, "text/html; charset=utf-8")
            if p == "/api/meta":
                return self._send(json.dumps({"hands": list(META["hands"]), "fps": META["fps"],
                                              "wrist_frame": META.get("wrist_frame"), "episodes": episodes()}).encode(),
                                  "application/json")
            if p.startswith("/api/ep/"):
                return self._send(json.dumps(episode(p[len("/api/ep/"):])).encode(), "application/json")
            if p.startswith("/urdf/"):
                hand, rel = p[len("/urdf/"):].split("/", 1)
                if rel == "robot.urdf":
                    return self._send(open(os.path.join(args.rt, "urdf", f"{hand}.urdf"), "rb").read(), "text/xml")
                f = resolve_mesh(hand, rel)
                if f is None:
                    return self._send(b"missing", "text/plain", 404)
                return self._send(open(f, "rb").read(), "application/octet-stream")
            return self._send(b"not found", "text/plain", 404)
        except Exception as e:  # noqa: BLE001
            return self._send(str(e).encode(), "text/plain", 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    mesh_index()   # build before serving
    print(f"serving {args.rt} on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
