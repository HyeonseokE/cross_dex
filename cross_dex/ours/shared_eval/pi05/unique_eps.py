"""Evaluation episodes with distinct initial layouts (ours, 2026-11-16).

Some merged teleop pickles contain several demos recorded from the same reset (same object / goal / table poses), e.g.
GraspBleach allegro episodes 0, 9, 12, 15. Evaluating on pickle episodes 0-29 then repeats layouts. This picks, for one
pickle, the requested episodes in order, drops any whose initial layout was already taken, and fills up from the remaining
episodes (ascending) until the requested count is reached. Pickles without repeats get the requested list unchanged.

Layout = every ``initial_state/rigid_object/*/root_pose`` rounded to 1e-4.

  python unique_eps.py <pickle> <episodes, e.g. 0,1,...,29>      -> prints the episode list (comma separated)
"""
import pickle
import sys

import numpy as np


def layout(ep):
    s = (ep.get("initial_state") or ep.get("init_state") or {}).get("rigid_object", {})
    return tuple(np.concatenate([np.round(np.asarray(s[k]["root_pose"], float).ravel(), 4) for k in sorted(s) if "root_pose" in s[k]]))


def unique_episodes(pkl, want):
    d = pickle.load(open(pkl, "rb"))
    eps = d["episodes"] if isinstance(d, dict) and "episodes" in d else d
    order = [i for i in want if i < len(eps)] + [i for i in range(len(eps)) if i not in set(want)]
    seen, out = set(), []
    for i in order:
        k = layout(eps[i])
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
        if len(out) == len(want):
            break
    if len(out) < len(want):
        sys.exit(f"{pkl}: only {len(out)} distinct initial layouts, {len(want)} wanted")
    return out


if __name__ == "__main__":
    print(",".join(map(str, unique_episodes(sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x]))))
