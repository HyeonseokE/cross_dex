"""LeRobot v3.0 reader for openpi training, so the XL-VLA latent datasets (dexverse_xlvla.make_dataset, <out>/all) are used
as they are, like the common LeRobot DP training (openpi's pinned lerobot reads only v2.1; openpi itself is not edited:
openpi_train.py swaps openpi.training.data_loader.create_torch_dataset for V3Dataset).

Sample i (global frame index) -> {"image", "wrist_image" (+ "left_wrist_image" on bimanual sets): HWC uint8 frames decoded from the AV1 videos at that timestamp,
"state": observation.state (41), "actions": action[i : i+H] (H, 41, last frame of the episode repeated past its end, as
lerobot's delta_timestamps clamping), "prompt": task sentence}.
Numeric columns are loaded once (small); videos are decoded per sample with torchcodec (one decoder cache per worker).

  python -m dexverse_xlvla.pi05.v3_dataset bench --root <make_dataset out>/all [--workers 8 --batch 32 --n 50]
"""
from __future__ import annotations

import argparse, collections, glob, json, os, time

import numpy as np
import pandas as pd

# DexVerse scripts/eval/baseline_tasks.py (verbatim prompts the DexVerse pi0.5 checkpoints were finetuned on)
PROMPTS = {
    "Dexverse-FunctionalHammerStrike-v0": "strike the nail with the hammer",
    "Dexverse-GraspBleach-v0": "grasp the bleach bottle",
    "Dexverse-GraspKettle-v0": "grasp the kettle",
    "Dexverse-GraspPan-v0": "grasp the pan",
    # bimanual (2026-11-16): same file's v0 sentences; folder / knife teleop sets are the v1 tasks
    "Dexverse-OpenFlatFolder-v1": "open the flat folder",
    "Dexverse-SlideUtilityKnife-v1": "slide out the utility knife blade",
    "Dexverse-BimanualLiftTray-v0": "lift the tray with both hands",
    "Dexverse-BimanualLiftCarton-v0": "lift the carton with both hands",
}
CAMS = {"image": "observation.images.third_person", "wrist_image": "observation.images.eye_in_hand"}
# bimanual sets (make_dataset_bimanual): eye_in_hand = right wrist camera, eye_in_hand_left = left wrist camera
CAMS_BIMANUAL = dict(CAMS, left_wrist_image="observation.images.eye_in_hand_left")


# task strings stored in the teleop LeRobot sets (meta/tasks.parquet) -> DexVerse env id
TELEOP_TASKS = {"Grasp bleach": "Dexverse-GraspBleach-v0", "Grasp pan": "Dexverse-GraspPan-v0",
                "Grasp kettle": "Dexverse-GraspKettle-v0", "Functional hammer strike": "Dexverse-FunctionalHammerStrike-v0",
                "Open flat folder": "Dexverse-OpenFlatFolder-v1", "Slide utility knife": "Dexverse-SlideUtilityKnife-v1",
                "Bimanual lift tray": "Dexverse-BimanualLiftTray-v0", "Bimanual lift carton": "Dexverse-BimanualLiftCarton-v0"}


def task_prompt(task: str) -> str:
    if task in TELEOP_TASKS:
        return PROMPTS[TELEOP_TASKS[task]]
    for k, v in PROMPTS.items():
        if k in task:
            return v
    raise KeyError(f"no prompt for task {task!r}")


class V3Dataset:
    def __init__(self, root: str, action_horizon: int, decoder_cache: int = 16):
        self.root, self.h, self.cache_n = root, action_horizon, decoder_cache
        self.info = json.load(open(f"{root}/meta/info.json"))
        self.cams = CAMS_BIMANUAL if CAMS_BIMANUAL["left_wrist_image"] in self.info["features"] else CAMS
        eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
        self.eps = eps.sort_values("episode_index").set_index("episode_index")
        cols = ["index", "episode_index", "timestamp", "task_index", "observation.state", "action"]
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
        df = df.sort_values("index").reset_index(drop=True)
        assert (df["index"].to_numpy() == np.arange(len(df))).all(), "global index is not 0..N-1"
        assert len(df) == self.info["total_frames"], (len(df), self.info["total_frames"])
        self.state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        self.action = np.stack(df["action"].to_numpy()).astype(np.float32)
        self.ep = df["episode_index"].to_numpy(np.int64)
        self.ts = df["timestamp"].to_numpy(np.float64)
        self.task = df["task_index"].to_numpy(np.int64)
        self.ep_end = self.eps["dataset_to_index"].to_numpy(np.int64)[np.searchsorted(self.eps.index.to_numpy(), self.ep)]
        tasks = pd.read_parquet(f"{root}/meta/tasks.parquet")      # index = task string, column task_index
        self.prompts = {int(r.task_index): task_prompt(t) for t, r in tasks.iterrows()}
        self._dec = collections.OrderedDict()

    def __len__(self):
        return len(self.state)

    def _decoder(self, path):
        from torchcodec.decoders import VideoDecoder
        d = self._dec.pop(path, None) or VideoDecoder(path)
        self._dec[path] = d
        while len(self._dec) > self.cache_n:
            self._dec.popitem(last=False)
        return d

    def frame(self, i: int, key: str) -> np.ndarray:
        vk = self.cams[key]; e = self.eps.loc[int(self.ep[i])]
        path = f"{self.root}/" + self.info["video_path"].format(video_key=vk, chunk_index=int(e[f"videos/{vk}/chunk_index"]),
                                                                file_index=int(e[f"videos/{vk}/file_index"]))
        t = float(e[f"videos/{vk}/from_timestamp"]) + float(self.ts[i])
        x = self._decoder(path).get_frames_played_at(seconds=[t]).data[0]     # (3, H, W) uint8
        return np.ascontiguousarray(x.numpy().transpose(1, 2, 0))

    def __getitem__(self, i):
        i = int(i)
        idx = np.minimum(np.arange(i, i + self.h), self.ep_end[i] - 1)
        out = {k: self.frame(i, k) for k in self.cams}
        out.update(state=self.state[i], actions=self.action[idx], prompt=self.prompts[int(self.task[i])])
        return out


def bench(a):
    import torch
    ds = V3Dataset(a.root, a.horizon)
    print(f"[bench] {len(ds)} frames, {len(ds.eps)} episodes, prompts {ds.prompts}", flush=True)
    x = ds[0]; print({k: (v.shape, v.dtype) if hasattr(v, "shape") else v for k, v in x.items()}, flush=True)
    t = time.time(); rng = np.random.default_rng(0)
    for i in rng.integers(0, len(ds), 50):
        ds[i]
    print(f"[bench] single process random access: {50 / (time.time() - t):.1f} samples/s", flush=True)
    for w in [int(v) for v in a.workers.split(",")]:
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=w, drop_last=True,
                                         persistent_workers=False, collate_fn=lambda b: b)
        it = iter(dl); next(it)                      # warm up (worker start)
        t = time.time()
        for _ in range(a.n):
            next(it)
        dt = time.time() - t
        print(f"[bench] workers {w}: {a.n * a.batch / dt:.1f} samples/s = {a.n / dt:.2f} batches/s (batch {a.batch})", flush=True)
        del it, dl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", default="4,8,16")
    ap.add_argument("--n", type=int, default=30)
    bench(ap.parse_args())
