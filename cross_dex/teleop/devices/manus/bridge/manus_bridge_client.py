"""Client for the Manus bridge (``manus_bridge_server.py``): keeps the latest glove data.

Standard library + numpy only, so it runs in any process (a simulator, a viewer, another PC) without
the Manus SDK or USB access. The server sends one JSON line per frame::

    {"v": 1, "t": <host time>, "seq": n,
     "gloves": [{"id": int, "side": "left"|"right", "age_s": float,
                 "kp": [75 floats],       # 25 nodes (x, y, z) m, WebXR order
                 "wq": [x, y, z, w],      # wrist-node quaternion
                 "ergo": {name: deg} | null}]}

    python manus_bridge_client.py [--host H] [--port P]     # print what arrives
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np

SIDES = ("right", "left")


class ManusBridgeClient:
    """Background TCP client for the Manus bridge; keeps only the latest message and reconnects."""

    def __init__(self, host: str = "127.0.0.1", port: int = 47000, stale_s: float = 0.5):
        self.host, self.port, self.stale_s = host, int(port), float(stale_s)
        self._lock = threading.Lock()
        self._msg: dict | None = None
        self._t = 0.0
        self.connected = False
        self.err = ""
        self.n = 0
        threading.Thread(target=self._run, daemon=True, name="manus-bridge-client").start()

    def _run(self) -> None:
        while True:
            try:
                with socket.create_connection((self.host, self.port), timeout=2.0) as s:
                    s.settimeout(2.0)
                    self.connected, self.err = True, ""
                    buf = b""
                    while True:
                        chunk = s.recv(65536)
                        if not chunk:
                            raise ConnectionError("bridge closed")
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        if lines:
                            msg = json.loads(lines[-1])  # latest line only
                            with self._lock:
                                self._msg, self._t = msg, time.monotonic()
                                self.n += len(lines)
            except Exception as exc:  # noqa: BLE001 - reconnect on any socket / parse error
                self.connected = False
                self.err = f"no Manus bridge at {self.host}:{self.port} ({type(exc).__name__})"
                time.sleep(1.0)

    def gloves(self) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        """``{side: (kp (25, 3), wrist quaternion xyzw or None)}`` for gloves seen within ``stale_s``."""
        with self._lock:
            msg, t = self._msg, self._t
        if msg is None or time.monotonic() - t > self.stale_s:
            return {}
        out = {}
        for g in msg.get("gloves", []):
            if g.get("age_s", 0.0) <= self.stale_s and g.get("side") in SIDES:
                wq = g.get("wq")
                out[g["side"]] = (
                    np.asarray(g["kp"], dtype=np.float64).reshape(25, 3),
                    None if wq is None else np.asarray(wq, dtype=np.float64),
                )
        return out

    def battery(self) -> dict[str, int | None]:
        """Battery % per streaming glove side (None until the SDK has reported it)."""
        with self._lock:
            msg = self._msg
        return {} if msg is None else {g["side"]: g.get("battery") for g in msg["gloves"]}

    def status(self) -> str:
        batt = self.battery()
        sides = [f"{s}:{batt[s]}%" if batt.get(s) is not None else s for s in sorted(self.gloves())]
        return f"bridge={'ok' if self.connected else self.err} msgs={self.n} gloves={sides}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    a = ap.parse_args()
    client = ManusBridgeClient(a.host, a.port)
    while True:
        time.sleep(1.0)
        gloves = client.gloves()
        tips = {s: np.round(kp[9], 3).tolist() for s, (kp, _) in gloves.items()}
        print(f"[client] {client.status()} index tip {tips}", flush=True)
