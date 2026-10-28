"""Six-hand retargeting viewer: one glove input -> every Bench2Dex hand, angle map and DexPilot side by side, in a
browser (three.js + urdf-loader from robo-viz/retarget_viewer/static/vendor). Nothing is sent to a robot.

    source /workspace/dexverse_env.sh && cd /workspace/DexVerse
    python vr_teleop/vr_teleop/tools/hand_viewer/server.py --source bridge --side right      # live glove (bridge must run)
    python vr_teleop/vr_teleop/tools/hand_viewer/server.py --source synth                    # open/fist/pinch cycle, no glove
    python vr_teleop/vr_teleop/tools/hand_viewer/server.py --source /workspace/.tmp/kp_right_live.npz   # replay

PC: ``ssh -L 8040:127.0.0.1:8040 bs1`` then http://127.0.0.1:8040 . Solid hand = angle map (angle_cfgs/<hand>.yml,
hot-reloaded; with the DexPilot thumb / pinch blend exactly as the teleop applies them), translucent blue = the
DexPilot solution seeded from the angle posture (hand_cfgs/<hand>.yml), i.e. what the pinch blend / fixed joints use. Hands without an angle config show DexPilot only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]  # .../DexVerse
sys.path.insert(0, str(REPO / "vr_teleop"))
sys.path.insert(0, str(REPO / "vr_teleop/vr_teleop/devices/manus/bridge"))

from vr_teleop.common.openxr_hand import HAND_JOINT_NAMES, PALM_DOWN_FINGERS_X, palm_down_frame  # noqa: E402
from vr_teleop.devices.manus.glove_skeleton import GloveWristFrame, glove_to_openxr_joints  # noqa: E402
from vr_teleop.teleop.retarget.angle_retargeter import ANGLE_CFGS_DIR, AngleHand  # noqa: E402
from vr_teleop.teleop.retarget.bench2dex_finger_retargeter import Bench2DexHand  # noqa: E402

HANDS = {  # viewer key -> (Bench2Dex hand_type, angle cfg name)
    "shadow": ("shadow", "shadow"), "allegro": ("allegro", "allegro"), "inspire": ("rh56dfx", "inspire"),
    "leap": ("leap", "leap"), "sharpa": ("sharpa", "sharpa"), "wuji": ("wuji", "wuji"),
}
URDF_ZOO = pathlib.Path(os.environ.get("URDF_ZOO_DIR", "/workspace/assets/bench2dex/URDF-Zoo"))
VENDOR = pathlib.Path(os.environ.get("HAND_VIEWER_VENDOR", "/workspace/robo-viz/retarget_viewer/static/vendor"))


def urdf_of(hand_type: str, side: str) -> pathlib.Path:
    import yaml

    cfg = yaml.safe_load(open(REPO / "vr_teleop/vr_teleop/teleop/retarget/bench2dex/hand_cfgs" / f"{hand_type}.yml"))
    return URDF_ZOO / cfg["retargeting"][side]["urdf_path"]


class HandRig:
    """One robot hand: DexPilot always, the angle map when a config exists (+ its DexPilot joints / pinch blend)."""

    def __init__(self, key: str, side: str):
        self.key, self.side = key, side
        hand_type, angle_name = HANDS[key]
        self.dex = Bench2DexHand(hand_type, side, input_device="manus")
        self.names = list(self.dex.joint_names)
        self.angle = None
        if (ANGLE_CFGS_DIR / f"{angle_name}.yml").is_file():
            self.angle = AngleHand(angle_name, side)
            self._fixed = [(self.angle.joint_names.index(n), self.names.index(n)) for n in self.angle.dexpilot_joints
                           if n in self.angle.joint_names and n in self.names]
            self._dp_index = [self.names.index(n) if n in self.names else -1 for n in self.angle.joint_names]
        self.urdf = urdf_of(hand_type, side)
        self.desc_root = self.urdf.parents[1] if self.urdf.parent.name == "urdf" else self.urdf.parent
        self.urdf_url = f"/desc/{key}/{self.urdf.relative_to(self.desc_root).as_posix()}"
        self.ms = 0.0

    def info(self) -> dict:
        wrist = getattr(self.dex.bridge, "wrist_link_name", "") or ""
        return {"key": self.key, "label": self.key, "urdf": self.urdf_url, "joints": self.names, "base_link": wrist,
                "angle": self.angle is not None,
                "pinch": list(self.angle.pinch_fingers) if self.angle else [], "fixed": list(self.angle.dexpilot_joints) if self.angle else []}

    def step(self, joints: dict) -> dict:
        t0 = time.perf_counter()
        out = {"q_dex": None, "q_angle": None, "blend": 0.0, "err": None}
        q = None
        if self.angle is not None:
            q = self.angle.retarget(joints)
            if q is not None:
                q = np.array(q, dtype=np.float32)
                self.dex.seed({n: float(v) for n, v in zip(self.angle.joint_names, q)})  # as the teleop does
        q_dex = self.dex.retarget(joints)
        if q_dex is not None:
            out["q_dex"] = {n: float(v) for n, v in zip(self.names, q_dex)}
        if self.angle is not None:
            if q is not None:
                if q_dex is not None:
                    w = self.angle.blend_weights(joints)
                    if w is not None:
                        out["blend"] = float(w.max())
                        for i, idp in enumerate(self._dp_index):
                            if idp >= 0 and w[i] > 0:
                                q[i] = (1.0 - w[i]) * q[i] + w[i] * q_dex[idp]
                    for ia, idp in self._fixed:
                        q[ia] = q_dex[idp]
                out["q_angle"] = {n: float(v) for n, v in zip(self.angle.joint_names, q)}
        self.ms = (time.perf_counter() - t0) * 1000
        out["ms"] = round(self.ms, 1)
        return out


class Source:
    """Glove keypoints (25, 3) + wrist quaternion from the bridge, a synthetic cycle, or an npz replay."""

    def __init__(self, spec: str, side: str):
        self.spec, self.side, self.err = spec, side, None
        self.frame_fn = GloveWristFrame()
        self.rot = PALM_DOWN_FINGERS_X if side == "right" else palm_down_frame(np.array([1.0, 0.0, 0.0]))
        self.t0 = time.monotonic()
        if spec == "bridge":
            from manus_bridge_client import ManusBridgeClient

            self.client = ManusBridgeClient("127.0.0.1", 47000)
        elif spec == "synth":
            from manus_bridge_server import fake_snapshot

            self.fake = fake_snapshot
        else:
            d = np.load(spec)
            self.rep = (d["t"] - d["t"][0], d["kp"], d["wq"] if "wq" in d else None)

    def kp(self):
        """(kp, wq) or None."""
        if self.spec == "bridge":
            g = self.client.gloves().get(self.side)
            self.err = None if g is not None else f"bridge: no {self.side} glove ({self.client.status()})"
            return None if g is None else (np.asarray(g[0], float), g[1])
        if self.spec == "synth":
            g = [g for g in self.fake(time.monotonic() - self.t0) if g["side"] == "right"]
            if not g:
                return None
            kp = np.asarray(g[0]["kp"], float).reshape(25, 3)
            if self.side == "left":
                kp = kp * np.array([1.0, -1.0, 1.0])
            return kp, g[0].get("wq")
        t, KP, WQ = self.rep
        i = int(np.searchsorted(t, (time.monotonic() - self.t0) % max(t[-1], 1e-3)))
        i = min(i, len(KP) - 1)
        return KP[i], (None if WQ is None else WQ[i])

    def joints(self):
        """OpenXR joints dict (wrist at the origin, palm-down frame) and the skeleton in that frame, or (None, None)."""
        got = self.kp()
        if got is None:
            return None, None
        kp, wq = got
        if not np.isfinite(kp).all():
            return None, None
        fr = self.frame_fn(kp, self.side, wq)
        if fr is None:
            return None, None
        joints = glove_to_openxr_joints(kp, fr, np.zeros(3), self.rot)
        skel = [[float(x) for x in np.asarray(joints[n])[:3]] for n in HAND_JOINT_NAMES]
        return joints, skel


def main():
    from aiohttp import web

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--source", default="bridge", help="bridge | synth | <recording.npz>")
    ap.add_argument("--hands", nargs="*", default=list(HANDS))
    ap.add_argument("--hz", type=float, default=30.0)
    a = ap.parse_args()

    rigs = {}
    for k in a.hands:
        try:
            rigs[k] = HandRig(k, a.side)
            print(f"[viewer] {k}: {rigs[k].urdf} angle={'yes' if rigs[k].angle else 'no'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[viewer] {k}: FAILED {type(e).__name__}: {e}", flush=True)
    src = Source(a.source, a.side)
    state = {"frame": None}

    def compute():
        joints, skel = src.joints()
        if joints is None:
            return {"type": "frame", "nokp": True, "err": src.err, "source": a.source, "side": a.side}
        hands = {}
        for k, rig in rigs.items():
            try:
                hands[k] = rig.step(joints)
            except Exception as e:  # noqa: BLE001
                hands[k] = {"q_dex": None, "q_angle": None, "err": f"{type(e).__name__}: {e}", "ms": 0.0, "blend": 0.0}
        return {"type": "frame", "skeleton": skel, "hands": hands, "err": src.err, "source": a.source, "side": a.side,
                "t": time.time()}

    async def producer():
        period = 1.0 / a.hz
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            try:
                state["frame"] = await loop.run_in_executor(None, compute)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            await asyncio.sleep(max(0.0, period - (loop.time() - t0)))

    async def index(_):
        return web.FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"})

    async def api_info(_):
        return web.json_response({"hands": [r.info() for r in rigs.values()], "side": a.side, "source": a.source,
                                  "joint_names": list(HAND_JOINT_NAMES)})

    async def desc(req):
        rig = rigs.get(req.match_info["hand"])
        if rig is None:
            raise web.HTTPNotFound()
        p = (rig.desc_root / req.match_info["path"]).resolve()
        if rig.desc_root.resolve() not in p.parents or not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def ws(req):
        sock = web.WebSocketResponse(heartbeat=10.0)
        await sock.prepare(req)
        try:
            while not sock.closed:
                if state["frame"] is not None:
                    await sock.send_str(json.dumps(state["frame"]))
                await asyncio.sleep(1.0 / a.hz)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return sock

    async def on_startup(app):
        app["producer"] = asyncio.create_task(producer())

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/info", api_info)
    app.router.add_get("/desc/{hand}/{path:.+}", desc)
    app.router.add_get("/ws", ws)
    app.router.add_static("/static/vendor", VENDOR)
    app.on_startup.append(on_startup)
    print(f"[viewer] http://{a.host}:{a.port}/  side={a.side} source={a.source} hands={list(rigs)}", flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
