"""Hold out (task, hand) cells of a merged pi0.5 training set (<out>/all of make_dataset / make_joint_dataset /
make_dataset_bimanual) without reconverting: the kept episodes' rows are copied, episode_index / index renumbered
0..E-1 / 0..N-1, the episode table rewritten (one data file); videos are hard-linked unchanged (episodes keep their
video file + from_timestamp, so V3Dataset decodes the same frames). Cell of an episode = position in
meta/xlvla_cells.json "sources" (merge order, `episodes` each), checked against the episode's task string.

Used when the per-cell sets are no longer on the training server (2026-11-19: bimanual joint diagonal on p1 from
xlvla_joint/pi05_bimanual_24/all; same cells left out as the pi05_XL bimanual diagonal set).

  python -m dexverse_xlvla.pi05.subset_v3 --src <all> --out <new all> --exclude <task>:<hand>,...
"""
import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", required=True, help="comma-separated <task>:<hand>")
    a = ap.parse_args()
    excl = [c for c in a.exclude.split(",") if c]
    cj = json.load(open(f"{a.src}/meta/xlvla_cells.json"))
    keys = list(cj["sources"])
    bad = set(excl) - set(keys)
    if bad:
        raise SystemExit(f"cells not in the set: {sorted(bad)}")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")

    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.src}/meta/episodes/*/*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    cell = np.repeat(np.arange(len(keys)), [cj["sources"][k]["episodes"] for k in keys])
    assert len(cell) == len(eps) and (eps["episode_index"].to_numpy() == np.arange(len(eps))).all()
    seen = {}   # each cell has one task string, and the task -> task string map is one-to-one
    for i, k in enumerate(keys):
        t = {str(list(x)[0]) for x in eps.loc[cell == i, "tasks"]}
        assert len(t) == 1, (k, t)
        seen.setdefault(cj["sources"][k]["task"], set()).update(t)
    assert all(len(v) == 1 for v in seen.values()) and len({next(iter(v)) for v in seen.values()}) == len(seen), seen
    keep = ~np.isin(cell, [keys.index(c) for c in excl])
    old_ids = eps.loc[keep, "episode_index"].to_numpy()
    remap = {int(o): n for n, o in enumerate(old_ids)}

    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.src}/data/*/*.parquet"))])
    data = data[data["episode_index"].isin(old_ids)].sort_values("index").reset_index(drop=True)
    data["episode_index"] = data["episode_index"].map(remap).astype(np.int64)
    data["index"] = np.arange(len(data), dtype=np.int64)

    ne = eps[keep].reset_index(drop=True).copy()
    ne["episode_index"] = np.arange(len(ne), dtype=np.int64)
    ln = ne["length"].to_numpy(np.int64)
    assert ln.sum() == len(data), (ln.sum(), len(data))
    ne["dataset_to_index"] = np.cumsum(ln)
    ne["dataset_from_index"] = ne["dataset_to_index"] - ln
    for c in ("data/chunk_index", "data/file_index", "meta/episodes/chunk_index", "meta/episodes/file_index"):
        ne[c] = 0
    # per-episode frame order unchanged -> each episode's rows are contiguous and match its new range
    first = data.groupby("episode_index")["index"].min().to_numpy()
    assert (first == ne["dataset_from_index"].to_numpy()).all()

    os.makedirs(f"{a.out}/data/chunk-000"); os.makedirs(f"{a.out}/meta/episodes/chunk-000")
    data.to_parquet(f"{a.out}/data/chunk-000/file-000.parquet", index=False)
    ne.to_parquet(f"{a.out}/meta/episodes/chunk-000/file-000.parquet", index=False)
    subprocess.run(["cp", "-al", f"{a.src}/videos", f"{a.out}/videos"], check=True)
    for f in ("tasks.parquet", "stats.json"):
        if os.path.exists(f"{a.src}/meta/{f}"):
            shutil.copy(f"{a.src}/meta/{f}", f"{a.out}/meta/{f}")
    info = json.load(open(f"{a.src}/meta/info.json"))
    info.update(total_episodes=len(ne), total_frames=int(len(data)), splits={"train": f"0:{len(ne)}"})
    json.dump(info, open(f"{a.out}/meta/info.json", "w"), indent=4)
    cj["excluded"] = sorted([c.split(":") for c in excl])
    cj["sources"] = {k: v for k, v in cj["sources"].items() if k not in excl}
    cj["cells"] = [c for c in cj["cells"] if ":".join(c) not in excl]
    cj["subset_of"] = os.path.abspath(a.src)
    json.dump(cj, open(f"{a.out}/meta/xlvla_cells.json", "w"), indent=1)
    print(f"kept {len(ne)}/{len(eps)} episodes, {len(data)} frames, excluded {excl} -> {a.out}")


if __name__ == "__main__":
    main()
