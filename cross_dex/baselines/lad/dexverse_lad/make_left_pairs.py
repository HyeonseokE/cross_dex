"""Left-hand LAD pairs from a right-hand pairs file made by build_pairs_bench.py (decision C: joint-wise mirror
conversion, no new retargeting). Same result as build_pairs_bench's left output; used where only the right file was
copied (the PC -> vast link is slow).

  python -m dexverse_lad.make_left_pairs <bench>_right_pairs.npy      # writes <bench>_left_pairs.npy (+ .meta.json)
"""
import json, sys

import numpy as np

from dexverse_lad.build_pairs_bench import left_map


def main():
    src = sys.argv[1]
    assert src.endswith("_right_pairs.npy")
    meta = json.load(open(src[:-4] + ".meta.json"))
    right = np.load(src, allow_pickle=True).item()
    out = {}
    for h, x in right.items():
        idx, sgn = left_map(meta["retarget_hand"][h], meta["joint_names"][h], meta["bench"])   # name based -> valid in any joint order
        out[h] = np.ascontiguousarray(x[:, idx] * sgn)
    dst = src.replace("_right_pairs.npy", "_left_pairs.npy")
    np.save(dst, out, allow_pickle=True)
    meta.update(side="left", left_conversion="right pairs * LEFT_MAP (dexverse_lad.build_pairs_bench)")
    json.dump(meta, open(dst[:-4] + ".meta.json", "w"), indent=1)
    print("wrote", dst, {h: v.shape for h, v in out.items()})


if __name__ == "__main__":
    main()
