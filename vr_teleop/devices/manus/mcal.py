"""Manus 장갑 캘리브레이션 관리 — 이름 붙여 저장해 두고, 쓸 때 골라 적용한다. 디폴트 이름 지정 가능.

캘리브레이션이 실제로 사는 곳 = SDK 설정 파일
`~/.config/Manus/Core 3/CoreLite.Settings.3.2.0.Calibrations.json` 의 `gloveProfiles[<장갑 ID>]`.
Manus 프로그램(브리지·텔레옵·sdk_client)이 시작할 때 여기서 장갑 ID 로 찾아 자동 적용하고, 끝날 때 다시 쓴다.
그래서 한 번 적용하면 다른 걸 적용하거나 새로 캘리브레이션(sdk_client `C`)할 때까지 그 장갑에 계속 쓰인다.

두 가지 저장 형식:
  프로필(.profile.json) — 설정 파일의 장갑별 항목 그대로. **장갑 꺼져 있어도 됨**. 양손 한 파일.
      python mcal.py snapshot --name hscho              # 지금 설정 파일의 캘리브레이션 → <dir>/hscho.profile.json
      python mcal.py apply                              # 디폴트 프로필 → 설정 파일 (Manus 프로그램 꺼져 있어야 함)
      python mcal.py apply --name eun --side right      # eun 프로필의 오른손만 적용
  .mcal — SDK `CoreSdk_GetGloveCalibration` 바이트 (Windows MANUS Core 의 .mcal 과 같은 형식). **장갑 켜져 있어야 함**.
      python mcal.py save --side right --name eun       # 지금 장갑에 적용된 값 → <dir>/eun_right.mcal
      python mcal.py load --side right [--name hscho]   # <dir>/hscho_right.mcal → 장갑 (이름 생략 = 디폴트)
      python mcal.py load --side right --file x.mcal    # Windows 에서 내보낸 파일 등
  python mcal.py list                                   # 저장된 것 + 디폴트
  python mcal.py default --name hscho                   # 디폴트 지정 (인자 없으면 현재 디폴트 출력)

저장 폴더: `MANUS_MCAL_DIR`, 없으면 `~/.config/Manus/mcal` (bs1 컨테이너에선 /workspace/manus_config/mcal).
SDK 위치: `MANUS_SDK_PY`, 없으면 /workspace/manus_sdk/Python → ~/manus_sdk/Python → ~/workspace/manus_sdk/Python.
컨테이너: `source /workspace/dexverse_env.sh` (cffi 있는 환경) 후
`python /workspace/DexVerse/vr_teleop/vr_teleop/devices/manus/mcal.py …` — 이 파일 하나로 동작한다(다른 모듈 불필요).
apply/save/load 는 동글을 쓰는 다른 Manus 프로그램(브리지 등)이 꺼져 있어야 한다 — apply 는 직접 확인하고 거부한다.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import glob
import json
import os
import shutil
import sys
import time

DEFAULT_DIR = os.environ.get("MANUS_MCAL_DIR") or os.path.expanduser("~/.config/Manus/mcal")
CONFIG_DIR = os.environ.get("MANUS_CONFIG_DIR") or os.path.expanduser("~/.config/Manus")
MANUS_VID = "3325"


def _default_name(d: str) -> str | None:
    try:
        with open(os.path.join(d, "DEFAULT")) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _name(args) -> str:
    name = args.name or _default_name(args.dir)
    if not name:
        sys.exit("--name 이 필요합니다 (디폴트 없음 — python mcal.py default --name <이름> 으로 지정)")
    return name


def _path(args) -> str:
    if getattr(args, "file", None):
        return os.path.abspath(args.file)
    return os.path.join(args.dir, f"{_name(args)}_{args.side}.mcal")


def _profile_path(d: str, name: str) -> str:
    return os.path.join(d, f"{name}.profile.json")


# ---------------------------------------------------------------- SDK 설정 파일
def _settings_path() -> str:
    hits = sorted(glob.glob(os.path.join(CONFIG_DIR, "Core*", "CoreLite.Settings.*.Calibrations.json")))
    if not hits:
        sys.exit(f"[mcal] 설정 파일 없음: {CONFIG_DIR}/Core*/CoreLite.Settings.*.Calibrations.json "
                 f"(Manus 프로그램을 한 번도 안 켰거나 MANUS_CONFIG_DIR 가 다름)")
    return hits[-1]


def _read_settings(path: str):
    """(헤더 바이트, dict) — 파일 앞에 `~<hex>~\\x03\\x02\\0\\0\\n` 헤더가 붙어 있고 그 뒤가 JSON."""
    with open(path, "rb") as f:
        raw = f.read()
    i = raw.index(b"{")
    return raw[:i], json.loads(raw[i:])


def _write_settings(path: str, head: bytes, d: dict) -> str:
    bak = f"{path}.bak_mcal_{dt.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(path, bak)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(json.dumps(d, indent=3, separators=(",", " : "), ensure_ascii=False).encode())
        f.write(b"\n")
    os.replace(tmp, path)
    return bak


def _dongle_holders() -> list[str]:
    """동글(USB 3325:*) 장치 파일을 열고 있는 프로세스 — 같은 PID 네임스페이스(컨테이너) 안만 보인다."""
    devs = set()
    for v in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            if open(v).read().strip() != MANUS_VID:
                continue
            base = os.path.dirname(v)
            bus, dev = int(open(f"{base}/busnum").read()), int(open(f"{base}/devnum").read())
            devs.add(f"/dev/bus/usb/{bus:03d}/{dev:03d}")
        except (OSError, ValueError):
            pass
    out = []
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) in devs:
                pid = fd.split("/")[2]
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
                out.append(f"PID {pid}: {cmd.strip()[:120]}")
        except OSError:
            pass
    return sorted(set(out))


def _glove_summary(gid: str, e: dict) -> str:
    kind = "Windows 프로필 포함" if "calibrationProfile" in e else "손가락 캘리브레이션만"
    return f"{gid} {e.get('side', '?'):5s} ({kind})"


# ---------------------------------------------------------------- 명령
def cmd_list(args):
    print(f"폴더: {args.dir}")
    print(f"디폴트: {_default_name(args.dir) or '(없음)'}")
    print(f"설정 파일: {_settings_path()}")
    if not os.path.isdir(args.dir):
        print("  (저장된 것 없음)")
        return
    for f in sorted(os.listdir(args.dir)):
        p = os.path.join(args.dir, f)
        t = dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
        if f.endswith(".profile.json"):
            try:
                with open(p) as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                continue
            sides = ",".join(e.get("side", "?") for e in meta.get("gloves", {}).values())
            note = f"  — {meta['note']}" if meta.get("note") else ""
            print(f"  [프로필] {f[:-13]:20s} {sides:11s} {t}{note}")
        elif f.endswith(".mcal"):
            meta = {}
            try:
                with open(p[:-5] + ".json") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                pass
            extra = f"  glove {meta['glove_id']}" if "glove_id" in meta else ""
            note = f"  — {meta['note']}" if meta.get("note") else ""
            print(f"  [.mcal]  {f:32s} {os.path.getsize(p):7d} B  {t}{extra}{note}")


def cmd_snapshot(args):
    if not args.name:
        sys.exit("--name 이 필요합니다")
    out = _profile_path(args.dir, args.name)
    if os.path.exists(out) and not args.force:
        sys.exit(f"[mcal] 이미 있음: {out} (덮어쓰려면 --force)")
    src = _settings_path()
    _, d = _read_settings(src)
    gloves = {gid: e for gid, e in d.get("gloveProfiles", {}).items() if args.side in (None, e.get("side"))}
    if not gloves:
        sys.exit(f"[mcal] 설정 파일에 {args.side or ''} 장갑 캘리브레이션이 없음: {src}")
    os.makedirs(args.dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"saved": dt.datetime.now().isoformat(timespec="seconds"), "note": args.note or "",
                   "source": src, "gloves": gloves}, f, ensure_ascii=False)
    print(f"[mcal] 저장: {out}")
    for gid, e in gloves.items():
        print("   ", _glove_summary(gid, e))


def cmd_apply(args):
    name = _name(args)
    src = _profile_path(args.dir, name)
    if not os.path.isfile(src):
        sys.exit(f"[mcal] 프로필 없음: {src}  (python mcal.py list 로 확인)")
    holders = _dongle_holders()
    if holders:
        sys.exit("[mcal] Manus 프로그램이 동글을 쓰는 중 — 끝날 때 설정 파일을 덮어쓰므로 먼저 끄세요:\n  "
                 + "\n  ".join(holders))
    with open(src) as f:
        gloves = {gid: e for gid, e in json.load(f)["gloves"].items() if args.side in (None, e.get("side"))}
    if not gloves:
        sys.exit(f"[mcal] {src} 에 {args.side} 장갑이 없음")
    path = _settings_path()
    head, d = _read_settings(path)
    d.setdefault("gloveProfiles", {}).update(gloves)
    bak = _write_settings(path, head, d)
    print(f"[mcal] 적용: {name} → {path}")
    for gid, e in gloves.items():
        print("   ", _glove_summary(gid, e))
    print(f"  이전 설정 백업: {bak}")
    print("  다음에 Manus 프로그램(브리지·텔레옵)을 켤 때부터 쓰인다.")


def cmd_default(args):
    if not args.name:
        print(_default_name(args.dir) or "(디폴트 없음)")
        return
    have = os.path.isfile(_profile_path(args.dir, args.name)) or \
        glob.glob(os.path.join(args.dir, f"{args.name}_*.mcal"))
    if not have:
        sys.exit(f"[mcal] '{args.name}' 프로필/.mcal 이 {args.dir} 에 없음")
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "DEFAULT"), "w") as f:
        f.write(args.name + "\n")
    print(f"[mcal] 디폴트 = {args.name}  (apply / load 에서 --name 생략 시 사용)")


def _sdk_path() -> str:
    env = os.environ.get("MANUS_SDK_PY")
    if env:
        return os.path.expanduser(env)
    for c in ("/workspace/manus_sdk/Python", "~/manus_sdk/Python", "~/workspace/manus_sdk/Python"):
        c = os.path.expanduser(c)
        if os.path.isdir(os.path.join(c, "manus_sdk")):
            return c
    sys.exit("[mcal] Manus SDK Python 폴더를 못 찾음 — MANUS_SDK_PY 로 지정")


def _open_glove(side: str, timeout: float):
    """SDK(Integrated) 시작 → `side` 장갑이 데이터를 보낼 때까지 기다려 (lib, ffi, glove_id).
    프로세스가 끝날 때 SDK 를 내린다(이때 SDK 가 설정 파일을 다시 쓴다)."""
    sys.path.insert(0, _sdk_path())
    try:
        from manus_sdk import ManusSDK, ffi, lib
        from manus_sdk.generated._enums import AxisPolarity, AxisView, SDKReturnCode, Side
    except ImportError as e:
        sys.exit(f"[mcal] Manus SDK import 실패: {e} — cffi 있는 환경인가? (source /workspace/dexverse_env.sh)")
    ok = SDKReturnCode.Success
    sdk = ManusSDK()
    live = set()                                     # 데이터를 보내고 있는 장갑 ID
    sdk.register_ergonomics_callback(
        lambda s: live.update(s.data[i].id for i in range(s.dataCount) if not s.data[i].isUserID))
    r = lib.CoreSdk_InitializeIntegrated()
    if r != ok:
        sys.exit(f"[mcal] SDK 초기화 실패: {SDKReturnCode(r).name} — 다른 Manus 프로그램(브리지 등)이 켜져 있나?")
    atexit.register(lambda: lib.CoreSdk_ShutDown())
    cs = ffi.new("CoordinateSystemVUH*")             # 연결 전에 좌표계를 정해야 데이터가 온다
    cs.view, cs.up, cs.handedness, cs.unitScale = AxisView.XFromViewer, AxisPolarity.PositiveZ, Side.Right, 1.0
    lib.CoreSdk_InitializeCoordinateSystemWithVUH(cs[0], True)
    t0, connected = time.time(), False
    while time.time() - t0 < timeout:
        if not connected and lib.CoreSdk_LookForHosts(1, False) == ok:
            cnt = ffi.new("uint32_t*")
            lib.CoreSdk_GetNumberOfAvailableHostsFound(cnt)
            if cnt[0]:
                hosts = ffi.new("ManusHost[]", cnt[0])
                lib.CoreSdk_GetAvailableHostsFound(hosts, cnt[0])
                connected = lib.CoreSdk_ConnectToHost(hosts[0]) == ok
        for gid in list(live):
            g = ffi.new("GloveLandscapeData*")
            if lib.CoreSdk_GetDataForGlove_UsingGloveId(gid, g) == ok and Side(g.side).name.lower() == side:
                return lib, ffi, gid
        time.sleep(0.2)
    sys.exit(f"[mcal] {side} 장갑이 {timeout:.0f}초 안에 안 보임 — 장갑 전원, 다른 Manus 프로그램(브리지 등) 종료 확인 "
             f"(동글 연결 {'됨' if connected else '안 됨'}, 데이터 오는 장갑 {[f'{g:#x}' for g in live]})")


def cmd_save(args):
    if not args.name and not args.file:
        sys.exit("--name 또는 --file 이 필요합니다")
    out = _path(args)
    if os.path.exists(out) and not args.force:
        sys.exit(f"[mcal] 이미 있음: {out} (덮어쓰려면 --force)")
    lib, ffi, gid = _open_glove(args.side, args.timeout)
    size = ffi.new("uint32_t*")
    r = lib.CoreSdk_GetGloveCalibrationSize(gid, size)
    if r != 0 or size[0] == 0:
        sys.exit(f"[mcal] 캘리브레이션 크기 읽기 실패 (code {r}, size {size[0]})")
    buf = ffi.new(f"unsigned char[{size[0]}]")
    r = lib.CoreSdk_GetGloveCalibration(buf, size[0])
    if r != 0:
        sys.exit(f"[mcal] 캘리브레이션 읽기 실패 (code {r})")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(bytes(ffi.buffer(buf, size[0])))
    with open(out[:-5] + ".json", "w") as f:
        json.dump({"glove_id": f"{gid:#x}", "side": args.side, "saved": dt.datetime.now().isoformat(timespec="seconds"),
                   "bytes": int(size[0]), "note": args.note or ""}, f, ensure_ascii=False, indent=1)
    print(f"[mcal] 저장: {out} ({size[0]} B, glove {gid:#x})")


def cmd_load(args):
    src = _path(args)
    if not os.path.isfile(src):
        sys.exit(f"[mcal] 파일 없음: {src}  (python mcal.py list 로 확인)")
    with open(src, "rb") as f:
        data = f.read()
    lib, ffi, gid = _open_glove(args.side, args.timeout)
    from manus_sdk.generated._enums import SetGloveCalibrationReturnCode
    buf = ffi.new(f"unsigned char[{len(data)}]")
    ffi.memmove(buf, data, len(data))
    res = ffi.new("SetGloveCalibrationReturnCode*")
    r = lib.CoreSdk_SetGloveCalibration(gid, buf, len(data), res)
    code = SetGloveCalibrationReturnCode(res[0]).name
    if r != 0 or code != "Success":
        sys.exit(f"[mcal] 적용 실패: SDK code {r}, result {code} "
                 f"(WrongSideError=손이 다름, VersionError/DeserializationError=다른 장갑 종류·버전 파일)")
    time.sleep(1.0)                      # 설정 파일 기록 여유
    print(f"[mcal] 적용: {src} → {args.side} 장갑 {gid:#x}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"저장 폴더 (기본 {DEFAULT_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("default", help="디폴트 이름 지정/출력")
    p.add_argument("--name")
    for name in ("snapshot", "apply"):
        p = sub.add_parser(name, help="설정 파일 → 프로필" if name == "snapshot" else "프로필 → 설정 파일")
        p.add_argument("--name", help="프로필 이름 → <dir>/<name>.profile.json (apply 에서 생략 = 디폴트)")
        p.add_argument("--side", choices=("left", "right"), help="한 손만 (기본: 양손)")
        if name == "snapshot":
            p.add_argument("--force", action="store_true", help="같은 이름 덮어쓰기")
            p.add_argument("--note", default="", help="메모(누구 손, 무슨 상황 등)")
    for name in ("save", "load"):
        p = sub.add_parser(name, help="장갑 → .mcal" if name == "save" else ".mcal → 장갑")
        p.add_argument("--side", choices=("left", "right"), required=True)
        p.add_argument("--name", help="이름 → <dir>/<name>_<side>.mcal (load 에서 생략 = 디폴트)")
        p.add_argument("--file", help="(load) 임의 경로의 .mcal / (save) 저장 경로 직접 지정")
        p.add_argument("--timeout", type=float, default=15.0, help="장갑을 기다릴 시간(초)")
        if name == "save":
            p.add_argument("--force", action="store_true", help="같은 이름 덮어쓰기")
            p.add_argument("--note", default="", help="메모(누구 손, 무슨 상황 등)")
    args = ap.parse_args()
    {"list": cmd_list, "default": cmd_default, "snapshot": cmd_snapshot, "apply": cmd_apply,
     "save": cmd_save, "load": cmd_load}[args.cmd](args)


if __name__ == "__main__":
    main()
