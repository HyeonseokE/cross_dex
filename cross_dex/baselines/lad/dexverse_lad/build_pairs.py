"""LAD paired action data (paper Eq. 1: x_i = (x^H_i, f^{R1}_H(x^H_i), ..., f^{RM}_H(x^H_i))) for the DexVerse hands.

The human hand poses x^H are DexCanvas trajectories; f^{Rj}_H is the DexVerse TELEOP retargeter of hand j (the function
the teleop demos were recorded with), run by cross_dex ``retarget_dexcanvas.py`` for all hands in ONE pass so that row i
of every hand comes from the same human frame. Its npz per trajectory has ``<hand>_q`` [T, n_finger] = finger targets
in env action order (= the finger slice of a teleop demo's ``action``).

Output: one ``.npy`` dict {hand: float32 (N, n_finger)} in the robot_clip data format (README: "dictionary with keys that
are the modality names and values that are arrays of shape (N, d_action)"), plus ``<out>.meta.json`` (joint names,
source trajectories, dropped rows). Only robot hands are modalities (paper Table II setup: robot end-effectors only; the
human pose is used to create the pairs).

  python -m dexverse_lad.build_pairs --src <retarget out dir> --out <pairs.npy> [--hands shadow,inspire,sharpa,wuji,allegro,leap]
"""
import argparse, glob, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji,allegro,leap")
    a = ap.parse_args()
    hands = a.hands.split(",")
    meta = json.load(open(f"{a.src}/meta.json"))
    files = sorted(glob.glob(f"{a.src}/*.npz"))
    cols = {h: [] for h in hands}
    used, dropped = [], 0
    for f in files:
        z = np.load(f)
        missing = [h for h in hands if f"{h}_q" not in z]
        if missing:
            raise SystemExit(f"{f}: no {missing} (retarget run must include all hands)")
        q = {h: np.asarray(z[f"{h}_q"], np.float32) for h in hands}
        T = {len(v) for v in q.values()}
        assert len(T) == 1, (f, {h: len(v) for h, v in q.items()})
        ok = np.all([np.isfinite(v).all(1) for v in q.values()], axis=0)
        dropped += int((~ok).sum())
        for h in hands:
            cols[h].append(q[h][ok])
        used.append(os.path.basename(f))
    data = {h: np.concatenate(v) for h, v in cols.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, data, allow_pickle=True)
    info = {"rows": int(len(next(iter(data.values())))), "dropped_nonfinite_rows": dropped, "trajectories": len(used),
            "dims": {h: int(v.shape[1]) for h, v in data.items()},
            "finger_joint_names": {h: meta["hands"][h]["finger_joint_names"] for h in hands if h in meta.get("hands", {})},
            "src": a.src, "files": used}
    json.dump(info, open(a.out + ".meta.json", "w"), indent=1)
    print({k: info[k] for k in ("rows", "dropped_nonfinite_rows", "trajectories", "dims")})


if __name__ == "__main__":
    main()
