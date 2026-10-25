"""Manus 장갑 입력 (Metagloves Pro, Manus SDK Integrated 모드) — 읽기 전용.

동글이 이 PC 에 꽂혀 있으면 MANUS Core 없이 SDK 가 장갑과 직접 통신한다(Integrated).
진동(haptics)·페어링·캘리브레이션 같은 쓰기 동작은 하지 않는다.

두 가지를 받는다:
  * raw skeleton 25 노드 — 노드 순서가 WebXR 25관절과 **같다**(2026-10-21 실측: 0 손목,
    엄지 1-4 Metacarpal·Proximal·Distal·Tip, 검지 5-9 … 새끼 20-24 Metacarpal·Proximal·
    Intermediate·Distal·Tip). 그래도 NodeInfo(chainType·fingerJointType)로 매번 자리를
    맞춰 넣는다 — 순서가 바뀌어도 조용히 틀리지 않게.
  * ergonomics 40 값(왼손 0-19, 오른손 20-39, °) — Tesollo 공식 Manus 매핑의 입력 그 자체.

SDK 는 프로세스당 한 번만 띄우고(뷰어가 꺼질 때 종료) 동글을 점유한다 → 이 뷰어가 Manus
입력을 쓰는 동안 다른 Manus 프로그램(예제 클라이언트·대시보드)은 끌 것.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

import numpy as np

SDK_PY = os.path.expanduser(os.environ.get("MANUS_SDK_PY", "~/workspace/manus_sdk/Python"))

# WebXR 인덱스: (chain, joint) → index
_CHAINS = {"FingerThumb": 1, "FingerIndex": 5, "FingerMiddle": 10, "FingerRing": 15, "FingerPinky": 20}
_THUMB_J = {"Metacarpal": 0, "Proximal": 1, "Distal": 2, "Tip": 3}
_FINGER_J = {"Metacarpal": 0, "Proximal": 1, "Intermediate": 2, "Distal": 3, "Tip": 4}

# ErgonomicsDataType 순서(손 하나 20개) → Tesollo manus_retarget.py 가 쓰던 이름.
# ⚠ Tesollo 예제는 구버전 이름(엄지 MCPSpread/MCPStretch/PIPStretch/DIPStretch = 지금의
# CMCSpread/CMCStretch/MCPStretch/IPStretch)을 쓴다 — 현 manus_ros2 이름(ThumbSpread·
# ThumbCMCStretch…)을 그대로 넘기면 예제의 ergo.get(…, 0.0) 이 엄지를 0 으로 읽는다.
TESOLLO_NAMES = (
    "ThumbMCPSpread", "ThumbMCPStretch", "ThumbPIPStretch", "ThumbDIPStretch",
    "IndexSpread", "IndexMCPStretch", "IndexPIPStretch", "IndexDIPStretch",
    "MiddleSpread", "MiddleMCPStretch", "MiddlePIPStretch", "MiddleDIPStretch",
    "RingSpread", "RingMCPStretch", "RingPIPStretch", "RingDIPStretch",
    "PinkySpread", "PinkyMCPStretch", "PinkyPIPStretch", "PinkyDIPStretch",
)


class ManusSource:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._users = 0
        self.err = ""
        self.connected = False
        self.gloves = {}        # glove id → {"side", "kp" (25,3), "ergo" [20], "t", "n"}
        self._nodeinfo = {}     # glove id → [(webxr index or None) per node]
        self.battery = {}       # glove id → 배터리 % (landscape 스트림)
        self._lib = self._ffi = None

    # --- lifecycle
    def acquire(self):
        with self._lock:
            self._users += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="manus")
                self._thread.start()

    def release(self):
        with self._lock:
            self._users = max(0, self._users - 1)   # SDK 는 띄운 채 둔다(재초기화 회피)

    def _run(self):
        try:
            sys.path.insert(0, SDK_PY)
            from manus_sdk import ManusSDK, ffi, lib
            from manus_sdk.generated._enums import (AxisPolarity, AxisView, HandMotion,
                                                    SDKReturnCode, Side)
        except Exception as e:
            self.err = f"Manus SDK import 실패({SDK_PY}): {type(e).__name__}: {e}"
            return
        self._lib, self._ffi, ok = lib, ffi, SDKReturnCode.Success
        self._sdk = ManusSDK()                      # 콜백 참조를 살려 둔다
        r = lib.CoreSdk_InitializeIntegrated()
        if r != ok:
            self.err = f"SDK 초기화 실패: {SDKReturnCode(r).name} — 다른 Manus 프로그램이 켜져 있나?"
            return
        atexit.register(lambda: lib.CoreSdk_ShutDown())
        self._sdk.register_raw_skeleton_callback(self._on_raw)
        self._sdk.register_ergonomics_callback(self._on_ergo)
        self._sdk.register_landscape_callback(self._on_landscape)
        cs = ffi.new("CoordinateSystemVUH*")
        cs.view, cs.up, cs.handedness, cs.unitScale = (AxisView.XFromViewer, AxisPolarity.PositiveZ,
                                                       Side.Right, 1.0)
        lib.CoreSdk_InitializeCoordinateSystemWithVUH(cs[0], True)     # 미터, 오른손 좌표계
        while not self.connected:
            if lib.CoreSdk_LookForHosts(1, False) == ok:
                cnt = ffi.new("uint32_t*")
                lib.CoreSdk_GetNumberOfAvailableHostsFound(cnt)
                if cnt[0]:
                    hosts = ffi.new("ManusHost[]", cnt[0])
                    lib.CoreSdk_GetAvailableHostsFound(hosts, cnt[0])
                    self.connected = lib.CoreSdk_ConnectToHost(hosts[0]) == ok
            if not self.connected:
                self.err = "동글 연결 대기 중…"
                time.sleep(1.0)
        self.err = ""
        lib.CoreSdk_SetRawSkeletonHandMotion(HandMotion.Auto)

    # --- callbacks (SDK 스레드)
    def _map_nodes(self, glove_id, n):
        from manus_sdk.generated._enums import ChainType, FingerJointType, Side
        ffi, lib = self._ffi, self._lib
        ni = ffi.new("NodeInfo[]", n)
        if lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, ni, n) != 0:
            return None, None
        idx, side = [], "right"
        for k in range(n):
            ch = ChainType(ni[k].chainType).name
            jt = FingerJointType(ni[k].fingerJointType).name
            side = "left" if Side(ni[k].side).name == "Left" else "right"
            if ch == "Hand":
                idx.append(0)
            elif ch in _CHAINS:
                table = _THUMB_J if ch == "FingerThumb" else _FINGER_J
                idx.append(_CHAINS[ch] + table[jt] if jt in table else None)
            else:
                idx.append(None)
        return idx, side

    def _on_raw(self, stream_info):
        ffi, lib = self._ffi, self._lib
        for i in range(stream_info.skeletonsCount):
            info = ffi.new("RawSkeletonInfo*")
            if lib.CoreSdk_GetRawSkeletonInfo(i, info) != 0:
                continue
            n, gid = info.nodesCount, info.gloveId
            nodes = ffi.new("SkeletonNode[]", n)
            if lib.CoreSdk_GetRawSkeletonData(i, nodes, n) != 0:
                continue
            if gid not in self._nodeinfo:
                self._nodeinfo[gid] = self._map_nodes(gid, n)
            idx, side = self._nodeinfo[gid]
            if idx is None:
                continue
            kp = np.full((25, 3), np.nan)
            wq = None
            for k in range(n):
                j = idx[k]
                if j is not None:
                    p = nodes[k].transform.position
                    kp[j] = (p.x, p.y, p.z)
                    if j == 0:                       # 손목 노드 방향 (x, y, z, w)
                        q = nodes[k].transform.rotation
                        wq = (q.x, q.y, q.z, q.w)
            with self._lock:
                g = self.gloves.setdefault(gid, {"side": side, "ergo": None, "n": 0})
                g.update(kp=kp, wq=wq, t=time.monotonic(), side=side)
                g["n"] += 1

    def _on_ergo(self, stream):
        for i in range(stream.dataCount):
            d = stream.data[i]
            vals = list(d.data)
            with self._lock:
                g = self.gloves.setdefault(d.id, {"side": None, "kp": None, "t": 0.0, "n": 0})
                g["ergo_all"] = vals

    def _on_landscape(self, landscape):
        d = landscape.gloveDevices
        with self._lock:
            self.battery = {d.gloves[i].id: int(d.gloves[i].batteryPercentage) for i in range(d.gloveCount)}

    # --- reader
    def latest(self, prefer: str = "right"):
        """(kp (25,3), ergo {Tesollo 이름: °}, side, glove id) — 가장 최근에 갱신된 장갑.
        `prefer` 쪽 손이 살아 있으면 그쪽."""
        now = time.monotonic()
        with self._lock:
            live = [(gid, g) for gid, g in self.gloves.items()
                    if g.get("kp") is not None and now - g.get("t", 0) < 0.5]
        if not live:
            return None
        live.sort(key=lambda x: (x[1]["side"] != prefer, -x[1]["t"]))
        gid, g = live[0]
        off = 0 if g["side"] == "left" else 20
        allv = g.get("ergo_all")
        ergo = None if allv is None else dict(zip(TESOLLO_NAMES, allv[off:off + 20]))
        return g["kp"].copy(), ergo, g["side"], gid

    def snapshot(self, max_age_s: float = 0.5) -> list:
        """살아 있는 장갑 전부 — [{"id", "side", "kp" (25,3), "ergo" {Tesollo 이름: °} | None,
        "age_s"}] (브리지가 내보내는 단위)."""
        now = time.monotonic()
        out = []
        with self._lock:
            for gid, g in self.gloves.items():
                if g.get("kp") is None or now - g.get("t", 0) > max_age_s:
                    continue
                off = 0 if g["side"] == "left" else 20
                allv = g.get("ergo_all")
                out.append({"id": gid, "side": g["side"], "kp": g["kp"].copy(), "wq": g.get("wq"),
                            "ergo": None if allv is None else dict(zip(TESOLLO_NAMES, allv[off:off + 20])),
                            "age_s": now - g["t"]})
        return out

    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            gl = [{"id": f"{gid:#x}", "side": g.get("side"), "n": g.get("n", 0),
                   "age_s": round(now - g["t"], 2) if g.get("t") else None}
                  for gid, g in self.gloves.items() if g.get("kp") is not None]
        return {"started": self._thread is not None, "connected": self.connected,
                "err": self.err, "gloves": gl}


MANUS = ManusSource()
