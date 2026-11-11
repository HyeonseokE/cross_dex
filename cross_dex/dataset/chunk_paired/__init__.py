"""Cross-embodiment paired chunk dataset spanning the grasp taxonomy (Feix GRASP, 33 types).

Human hand motion (DexCanvas, HOGraspNet) and 5-finger robot grasps (Dexonomy) are put into one representation,
labelled with Feix grasp types, retargeted to every 5-finger hand with the DexVerse teleop retargeter itself,
resampled to the policy rate and cut into H-step chunks that pair across hands (same source motion, same time window).
Construction steps and design: RSS2027 bundle ``ablation/dexverse-chunk-latent/DATASET_PIPELINE.md``.
"""

import os as _os

# Generated data (per source: dexcanvas/, hograspnet/, dexonomy/, transition/). On servers this is a link to
# /workspace/datasets/cross_dex/chunk_paired_dataset (data stays out of the synced code copy, git-ignored).
DATA_ROOT = _os.environ.get("CHUNK_PAIRED_DATA", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "chunk_paired_dataset"))


def save_manifest(out: str, args, **extra) -> str:
    """Run record next to a generated output: <out>/manifest.json for a folder, <out>.manifest.json for a file.

    command, arguments, start/end time (KST), host, git commits of cross_dex and DexVerse (the retargeter), plus counts.
    """
    import datetime as _dt
    import json as _json
    import socket as _socket
    import subprocess as _sp
    import sys as _sys

    def _git(d):
        try:
            c = _sp.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            dirty = bool(_sp.run(["git", "-C", d, "status", "--porcelain", "-uno"], capture_output=True, text=True,
                                 timeout=30).stdout.strip())
            return {"commit": c or None, "dirty": dirty}
        except Exception:  # noqa: BLE001
            return {"commit": None, "dirty": None}

    here = _os.path.dirname(_os.path.abspath(__file__))
    codes = _os.path.abspath(_os.path.join(here, "..", "..", "..", ".."))
    kst = _dt.timezone(_dt.timedelta(hours=9))
    rec = {"command": " ".join(_sys.argv), "python": _sys.executable, "args": vars(args) if args is not None else None,
           "end_time": _dt.datetime.now(kst).isoformat(timespec="seconds"), "host": _socket.gethostname(),
           "server": _os.environ.get("SERVER_NAME") or next((d for d in sorted(_os.listdir("/workspace/experiments"))
                                                            if not d.startswith(("_", ".")) and _os.path.isdir(
                                                                _os.path.join("/workspace/experiments", d))), None)
           if _os.path.isdir("/workspace/experiments") else None,
           "git": {"cross_dex": _git(_os.path.join(codes, "cross_dex")),
                   "DexVerse": _git(_os.path.join(codes, "dexverse", "DexVerse"))}}
    rec.update(extra)
    path = _os.path.join(out, "manifest.json") if _os.path.isdir(out) else out + ".manifest.json"
    with open(path, "w") as f:
        _json.dump(rec, f, indent=1, default=str)
    return path
