"""Manus 장갑 브리지 — 호스트에서 동글을 잡고(SDK Integrated, 읽기만) 장갑 값을 TCP 로 내보낸다.

동글은 한 프로그램만 쓸 수 있으므로 이 브리지 **하나만** SDK 를 띄우고, 나머지(뷰어, `--network host`
컨테이너의 시뮬레이터 등)는 `client.py` 로 받기만 한다. 받는 쪽엔 Manus SDK·USB 권한이 필요 없다.

    python bridge.py                     # 127.0.0.1:47000, 최대 100 Hz
    python bridge.py --bind 0.0.0.0      # 다른 PC(예: Tailscale)에서도 받게 — 인증 없음, 신뢰망에서만
    python bridge.py --fake              # 장갑 없이 합성 손으로 배관 점검

프로토콜: 연결하면 서버가 줄 단위 JSON 을 계속 보낸다(클라이언트는 아무것도 안 보냄).
  {"v": 1, "t": <호스트 time.time()>, "seq": n,
   "gloves": [{"id": int, "side": "left"|"right", "age_s": float,
               "kp": [75 floats]  # WebXR 25관절 (x,y,z) m — 0 손목, 엄지 1-4, 검지 5-9 … 새끼 20-24
               "ergo": {"ThumbMCPSpread": °, … 20개} | null}]}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_snapshot(t: float) -> list:
    """장갑 없이 쓰는 합성 오른손(굽힘이 천천히 오르내림) — 배관 점검용."""
    curl = 35.0 * (1 - math.cos(t * 1.2))          # 0..70°
    kp = np.zeros((25, 3))
    for b, x in ((5, 0.024), (10, 0.002), (15, -0.018), (20, -0.036)):
        kp[b] = (x * 0.6, 0.012, 0.0)
        p = np.array([x, 0.066, 0.0])
        kp[b + 1] = p
        a = 0.0
        for k, L in enumerate((0.040, 0.025, 0.020)):
            a += math.radians(curl)
            p = p + L * np.array([0.0, math.cos(a), math.sin(a)])
            kp[b + 2 + k] = p
    kp[1:5] = [(0.030, 0.010, 0.005), (0.050, 0.040, 0.012), (0.065, 0.065, 0.02), (0.075, 0.085, 0.025)]
    names = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    ergo = {}
    for n in names:
        if n == "Thumb":
            ergo.update(ThumbMCPSpread=10.0, ThumbMCPStretch=20.0, ThumbPIPStretch=curl * 0.3,
                        ThumbDIPStretch=curl * 0.3)
        else:
            ergo.update({f"{n}Spread": 0.0, f"{n}MCPStretch": curl, f"{n}PIPStretch": curl,
                         f"{n}DIPStretch": curl})
    return [{"id": 0, "side": "right", "kp": kp, "ergo": ergo, "age_s": 0.0}]


class Server:
    def __init__(self, bind: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.listen(8)
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            c, addr = self.sock.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c.settimeout(0.05)                       # 느린 클라이언트가 브리지를 막지 않게
            with self.lock:
                self.clients.append(c)
            print(f"[bridge] client {addr[0]}:{addr[1]} connected ({len(self.clients)})", flush=True)

    def send(self, line: bytes):
        with self.lock:
            dead = []
            for c in self.clients:
                try:
                    c.sendall(line)
                except (socket.timeout, OSError):
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass
                print(f"[bridge] client dropped ({len(self.clients)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--fake", action="store_true", help="장갑 없이 합성 손 송출")
    a = ap.parse_args()

    reader = None
    if not a.fake:
        sys.path.insert(0, HERE)
        from manus_sdk_reader import MANUS
        reader = MANUS
        reader.acquire()
    srv = Server(a.bind, a.port)
    print(f"[bridge] {'FAKE ' if a.fake else ''}listening on {a.bind}:{a.port} @ {a.hz:g} Hz", flush=True)

    seq, t0, last_log = 0, time.monotonic(), 0.0
    period = 1.0 / a.hz
    while True:
        t = time.monotonic()
        gloves = fake_snapshot(t - t0) if a.fake else reader.snapshot()
        msg = {"v": 1, "t": time.time(), "seq": seq,
               "gloves": [{"id": int(g["id"]), "side": g["side"], "age_s": round(g["age_s"], 3),
                           "kp": [round(float(v), 5) for v in np.asarray(g["kp"]).reshape(-1)],
                           "wq": None if g.get("wq") is None else [round(float(v), 6) for v in g["wq"]],
                           "ergo": None if g["ergo"] is None else
                           {k: round(float(v), 2) for k, v in g["ergo"].items()}}
                          for g in gloves]}
        srv.send((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        seq += 1
        if t - last_log > 5.0:
            last_log = t
            st = "" if a.fake else f" sdk={'ok' if reader.connected else (reader.err or 'starting')}"
            battery = {} if a.fake else reader.battery  # glove id -> % (SDK landscape)
            listed = [(hex(g['id']), g['side'], f"{battery[g['id']]}%" if g['id'] in battery else '?%') for g in gloves]
            print(f"[bridge] seq={seq} gloves={listed} "
                  f"clients={len(srv.clients)}{st}", flush=True)
        time.sleep(max(0.0, period - (time.monotonic() - t)))


if __name__ == "__main__":
    main()
