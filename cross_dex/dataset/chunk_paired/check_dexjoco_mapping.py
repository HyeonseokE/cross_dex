"""Check the DexVerse/retarget -> DexJoCo-X actuator mapping (build_chunks.BENCH["dexjoco"]) in the DexJoCo-X MuJoCo model.

For sampled retargeted frames of each hand: the retargeter's own fingertip positions (URDF FK, <hand>_links) vs the
fingertips of the DexJoCo-X model driven by the mapped actuator targets (MuJoCo, gravity off, the hand settles on its
position actuators, mimic joints follow their equality constraints). Fingertip-to-fingertip distances do not depend on
the frame, so they are compared directly; the DexJoCo fingertip bodies are found automatically (the leaf bodies of the hand
subtree, best matching assignment). A correct mapping gives a small distance error; for contrast the same check is run
with deliberately wrong mappings (identity order, flipped abduction). Also renders a few poses (PNG) for a visual check.

  MUJOCO_GL=egl /workspace/envs/dexjoco/main/bin/python -m cross_dex.dataset.chunk_paired.check_dexjoco_mapping --out <dir>
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

from cross_dex.dataset.chunk_paired.build_chunks import BENCH

D = "/workspace/datasets/cross_dex/chunk_paired_dataset"
DJ = os.environ.get("DEXJOCO_ROOT", "/workspace/codes/dexjoco/DexJoCo-X")
DJ_NAME = {"allegro": "allegro", "inspire": "inspire", "leap": "leap", "sharpa": "sharpa_wave", "wuji": "wuji",
           "linkerhand": "linkerhand_l20", "xhand": "xhand"}


def retarget_samples(rt_hand, n, rng):
    run = "feix400_v2" if rt_hand in ("shadow", "inspire", "sharpa", "wuji") else "feix400_v2_x"
    root = f"{D}/hograspnet/retarget/{run}"
    meta = json.load(open(f"{root}/meta.json"))["hands"][rt_hand]
    names = meta["link_names"]
    tips = list(dict.fromkeys(meta["task_links"]))
    tip_i = [names.index(t) for t in tips]
    files = sorted(glob.glob(f"{root}/*.npz"))
    Q, P = [], []
    for f in rng.choice(files, n, replace=False):
        z = np.load(f)
        t = rng.integers(len(z["t"]))
        Q.append(z[f"{rt_hand}_q"][t]); P.append(z[f"{rt_hand}_links"][t][tip_i])
    return np.asarray(Q, np.float64), np.asarray(P, np.float64), tips, meta["finger_joint_names"]


class DJHand:
    def __init__(self, dj_name):
        import mujoco
        self.mj = mujoco
        xml = f"{DJ}/simulator/dexjoco/sim/envs/xmls/benchmark7_v3_pick_bucket_{dj_name}.xml"
        con = json.load(open(f"{DJ}/evaluation/contracts/pick_bucket/{dj_name}.json"))
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.gravity[:] = 0
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # kinematic check: no self / scene contact
        self.d = mujoco.MjData(self.m)
        self.act = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in con["actuator_names"]["right"]]
        assert min(self.act) >= 0, "actuator names not found"
        hj = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in con["joint_names"]["right"]]
        hb = {int(self.m.jnt_bodyid[j]) for j in hj if j >= 0}
        par = self.m.body_parentid

        def anc(b):
            out = [b]
            while b != 0:
                b = int(par[b]); out.append(b)
            return out
        common = set(anc(next(iter(hb))))
        for b in hb:
            common &= set(anc(b))
        root = max(common, key=lambda b: len(anc(b)))          # lowest common ancestor of the hand joints
        sub = [b for b in range(self.m.nbody) if root in anc(b) and b != root]
        has_child = {int(par[b]) for b in sub}
        self.leaves = [b for b in sub if b not in has_child]
        self.leaf_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.leaves]
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.home = self.d.qpos.copy(); self.ctrl0 = self.d.ctrl.copy()
        self.lo, self.hi = self.m.actuator_ctrlrange[self.act].T
        # fingertip point per leaf: some models put the "tip" body at the last joint (allegro, leap) -> use the geom of the
        # leaf body (or its parent) farthest from the hand root at the home pose
        self.tip_geom = []
        for b in self.leaves:
            gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in (b, int(par[b]))]
            self.tip_geom.append(max(gs, key=lambda g: np.linalg.norm(self.d.geom_xpos[g] - self.d.xpos[root])) if gs else -1)
        self.root = root

    def tips(self, q, steps=400):
        self.d.qpos[:] = self.home; self.d.qvel[:] = 0; self.d.ctrl[:] = self.ctrl0
        self.d.ctrl[self.act] = np.clip(q, self.lo, self.hi)
        for _ in range(steps):
            self.mj.mj_step(self.m, self.d)
        return np.stack([self.d.geom_xpos[g] if g >= 0 else self.d.xpos[b] for g, b in zip(self.tip_geom, self.leaves)])


def _anc(m, b):
    out = [b]
    while b != 0:
        b = int(m.body_parentid[b]); out.append(b)
    return out


def pdist(P):  # [N, m, 3] -> [N, m, m]
    return np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)


# alternative (suspected) mappings, applied on top of the mapped DexJoCo-order vector
def _swap_leap(q):
    q = q.copy()
    for f in range(3):
        q[[4 * f, 4 * f + 1]] = q[[4 * f + 1, 4 * f]]
    return q


def _flip(idx):
    def f(q):
        q = q.copy(); q[idx] = -q[idx]; return q
    return f


ALT = {"leap": {"swap_abd_mcp": _swap_leap},
       "sharpa": {"flip_ring_pinky_AA": _flip([14, 19]), "flip_all_AA": _flip([6, 10, 14, 19])},
       "allegro": {"flip_abd": _flip([0, 4, 8])},
       "wuji": {"flip_J2": _flip([5, 9, 13, 17])}}


def err_of(Ddj, tup, Drt):
    t = list(tup)
    X = Ddj[:, t][:, :, t]
    iu = np.triu_indices(Drt.shape[1], 1)
    a, b = X[:, iu[0], iu[1]], Drt[:, iu[0], iu[1]]
    return round(float(np.abs(a - b).mean()) * 1e3, 2), round(float(np.abs((a - a.mean(0)) - (b - b.mean(0))).mean()) * 1e3, 2)


def best_assignment(Drt, Ddj):
    """Drt [N,m,m] retarget tips, Ddj [N,L,L] DexJoCo leaves -> (error m, tuple of leaf idx)"""
    m, L = Drt.shape[1], Ddj.shape[1]
    best = (np.inf, None)
    iu = np.triu_indices(m, 1)
    for tup in itertools.permutations(range(L), m):
        e = np.abs(Ddj[:, tup][:, :, tup][:, iu[0], iu[1]] - Drt[:, iu[0], iu[1]]).mean()
        if e < best[0]:
            best = (e, tup)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="allegro,inspire,leap,sharpa,wuji,linkerhand,xhand")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = BENCH["dexjoco"]
    rng = np.random.default_rng(0)
    report = {}
    for h in a.hands.split(","):
        rt = B["hands"][h]
        Q, Prt, tip_names, jn = retarget_samples(rt, a.n, rng)
        order = np.asarray(B["order"].get(h, range(Q.shape[1])))
        hand = DJHand(DJ_NAME[h])
        assert len(hand.act) == len(order), f"{h}: {len(hand.act)} actuators vs {len(order)} mapped joints"
        Drt = pdist(Prt)

        def run(qs):
            return pdist(np.stack([hand.tips(q) for q in qs]))
        Dmap = run(Q[:, order])
        err, tup = best_assignment(Drt, Dmap)
        r = {"tips_retarget": tip_names, "tips_dexjoco": [hand.leaf_names[i] for i in tup],
             "mapped_err_mm(abs,centered)": err_of(Dmap, tup, Drt),
             "tip_dist_mean_mm": round(float(Drt[:, *np.triu_indices(Drt.shape[1], 1)].mean()) * 1e3, 1)}
        # contrast: identity order (if different), every 'abduction-like' joint (name has AA / abd / roll / J2 / _0) sign-flipped
        if not np.array_equal(order, np.arange(len(order))):
            r["identity_order_err_mm"] = err_of(run(Q), tup, Drt)
        for k, f in ALT.get(h, {}).items():
            r[f"alt_{k}_err_mm"] = err_of(run(np.stack([f(q) for q in Q[:, order]])), tup, Drt)
        r["shuffled_pose_err_mm"] = err_of(run(Q[:, order][rng.permutation(len(Q))]), tup, Drt)   # unrelated poses: chance
        report[h] = r
        print(h, json.dumps(r), flush=True)
        # renders: 4 poses x 3 views, camera on the palm (hand root), arm hidden
        try:
            import mujoco
            from PIL import Image
            ren = mujoco.Renderer(hand.m, 360, 360)
            opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
            sub = {b for b in range(hand.m.nbody) if hand.root in _anc(hand.m, b)}
            for g in range(hand.m.ngeom):
                if hand.m.geom_bodyid[g] not in sub:
                    hand.m.geom_rgba[g, 3] = 0.0
            cam = mujoco.MjvCamera(); cam.distance = 0.32
            rows = []
            for q in Q[:4][:, order]:
                hand.tips(q)
                cam.lookat[:] = hand.d.xpos[hand.root] + 0.5 * (hand.d.xpos[hand.leaves].mean(0) - hand.d.xpos[hand.root])
                ims = []
                for az, el in ((0, -20), (90, -20), (180, -60)):
                    cam.azimuth, cam.elevation = az, el
                    ren.update_scene(hand.d, cam, opt); ims.append(ren.render())
                rows.append(np.concatenate(ims, 1))
            Image.fromarray(np.concatenate(rows, 0)).save(f"{a.out}/{h}.png")
        except Exception as e:  # rendering is optional
            print("render failed:", repr(e)[:200])
    json.dump(report, open(f"{a.out}/mapping_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
