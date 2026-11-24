"""Contact list -> fixed-size arrays (shared by dexo_extract_full.py and contact_match_grid.py)."""
import numpy as np

SEC = 8


def contacts_arrays(cons):   # (finger, distal, pos, dist, body) list -> pc, pm, sc, sm
    pc = np.zeros((5, 3), np.float32); pm = np.zeros(5, np.float32); sc = np.zeros((5, SEC, 3), np.float32); sm = np.zeros((5, SEC), np.float32)
    best = {}; sec = {f: [] for f in range(5)}
    for fi, distal, pos, dist, _ in cons:
        if fi >= 5:
            continue
        if distal:
            if fi not in best or dist < best[fi][1]:
                best[fi] = (pos, dist)
        else:
            sec[fi].append(pos)
    for f, (p, _) in best.items():
        pc[f] = p; pm[f] = 1
    for f, v in sec.items():
        v = v[::max(1, len(v) // SEC)][:SEC]
        for k, p in enumerate(v):
            sc[f, k] = p; sm[f, k] = 1
    return pc, pm, sc, sm
