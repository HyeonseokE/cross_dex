"""Trajectory pickle for an open-loop replay through the conversion path (record_demos --replay_demos).

Keeps the chosen episodes of a hand's own teleop pickle unchanged (initial state, goal, metadata incl. the wrist camera
body collision) and, in mode shared, swaps the finger actions for the ones in <actions.npz> (dexverse_xlvla.replay,
self replay: fingers D_h(E_h(q*)), wrist copied). Mode raw writes the recorded actions as they are.
Run with the simulator env (numpy 1.x, the env that wrote the pickles).

  python make_replay_pkl.py <src.pkl> <raw|shared|fixed|cross> <episodes, e.g. 0-9> <out.pkl> [actions.npz]
"""
import pickle
import sys

import numpy as np


def main():
    src, mode, spec, out = sys.argv[1:5]
    a, _, b = spec.partition("-")
    eps = list(range(int(a), int(b) + 1)) if b else [int(x) for x in spec.split(",")]
    d = pickle.load(open(src, "rb"))
    npz = np.load(sys.argv[5]) if mode != "raw" else None
    new = []
    for k in eps:
        e = dict(d["episodes"][k])
        if mode == "cross":   # <src.pkl> = cross_build's demos.pkl (t's pickle, s's scene): whole action = npz (t order)
            e["actions"] = npz[str(k)].astype(np.asarray(e["actions"]).dtype)
            print(f"episode {k}: cross actions {e['actions'].shape}")
        elif mode != "raw":
            act = np.asarray(e["actions"]).copy()
            sh = npz[str(k)]
            assert sh.shape == act.shape, (sh.shape, act.shape)
            assert np.array_equal(sh[:, :6], act[:, :6].astype(sh.dtype)), "wrist must be unchanged"
            cols = [c for c in range(6, act.shape[1]) if np.any(sh[:, c] != 0)]   # finger joints written by replay.py
            act[:, cols] = sh[:, cols].astype(act.dtype)
            e["actions"] = act
            print(f"episode {k}: {len(cols)}/{act.shape[1] - 6} finger columns replaced, "
                  f"mean |change| {np.abs(act - np.asarray(d['episodes'][k]['actions'])).mean():.4f}")
        new.append(e)
    d = dict(d)
    d["episodes"], d["num_episodes"] = new, len(new)
    d["replay_source"] = {"pickle": src, "mode": mode, "episodes": eps}
    pickle.dump(d, open(out, "wb"))
    print(out, len(new), "episodes")


if __name__ == "__main__":
    main()
