"""Finger-finger interpenetration of retargeted hand trajectories (quality metric: physically infeasible frames).

The teleop retargeter (DexPilot) has no collision term, so retargeted fingers can pass through each other. Per hand the
retarget URDF (as used by the optimizer, ``<rt>/urdf/<hand>.urdf``) is loaded with its collision meshes into pinocchio
+ hpp-fcl; every pair of collision geometries on two different fingers (first joint below the hand root differs) is
checked (collision test with contact depth), palm geometries are skipped.

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/main/bin/python \
      -m cross_dex.dataset.chunk_paired.self_collision --rt <retarget out dir> [--traj traj_00003931] [--out report.json]

Collision meshes missing next to a URDF are looked up by file name under --mesh_roots (as the web viewer does).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pinocchio as pin

from cross_dex.dataset.chunk_paired import save_manifest

try:
    import coal as fcl
except ImportError:  # older pinocchio
    import hppfcl as fcl


def _mesh_index(roots: list[str]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for root in roots:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".stl", ".obj", ".dae")):
                    idx.setdefault(f, []).append(os.path.join(dp, f))
    return idx


def resolved_urdf(hand: str, urdf: str, urdf_orig: str, index: dict[str, list[str]], tmp: str) -> str:
    """Copy of the URDF with every mesh filename made absolute (next to the original URDF, or found by name)."""
    udir = os.path.dirname(urdf_orig)
    text = open(urdf).read()

    def fix(m):
        rel = m.group(1)
        for c in [os.path.join(udir, rel)] + [os.path.join(d, rel) for d in glob.glob(os.path.join(udir, "*/"))]:
            if os.path.isfile(c):
                return f'filename="{c}"'
        hits = index.get(os.path.basename(rel), [])
        hits = [h for h in hits if hand in h.lower()] or hits
        return f'filename="{hits[0]}"' if hits else m.group(0)

    out = os.path.join(tmp, f"{hand}.urdf")
    open(out, "w").write(re.sub(r'filename="([^"]+)"', fix, text))
    return out


class HandCollision:
    def __init__(self, urdf: str, dex_joint_names: list[str]):
        self.model = pin.buildModelFromUrdf(urdf)
        self.geom = pin.buildGeomFromUrdf(self.model, urdf, pin.GeometryType.COLLISION)
        for g in self.geom.geometryObjects:   # convex hulls: GJK/EPA gives a real penetration depth (BVH meshes do not)
            if isinstance(g.geometry, fcl.BVHModelBase):
                g.geometry.buildConvexRepresentation(False)
                g.geometry = g.geometry.convex
        # finger of a geometry = the joint right below the deepest joint shared by all geometries (wrist joints of a
        # floating URDF are shared by every finger); geometries on that shared chain are the palm (-> 0, not checked)
        sups = [list(self.model.supports[g.parentJoint]) for g in self.geom.geometryObjects]
        common = 0
        while all(len(s) > common + 1 for s in sups if len(s) > common) and len({s[common] for s in sups if len(s) > common}) == 1 \
                and len({s[common + 1] for s in sups if len(s) > common + 1}) == 1:
            common += 1
        finger = [s[common + 1] if len(s) > common + 1 else 0 for s in sups]
        self.pairs = [(a, b) for a in range(len(finger)) for b in range(a + 1, len(finger))
                      if finger[a] and finger[b] and finger[a] != finger[b]]
        for a, b in self.pairs:
            self.geom.addCollisionPair(pin.CollisionPair(a, b))
        self.data, self.gdata = self.model.createData(), pin.GeometryData(self.geom)
        self.q_index = [self.model.joints[self.model.getJointId(n)].idx_q for n in dex_joint_names]
        self.names = [g.name for g in self.geom.geometryObjects]
        assert self.pairs, f"no finger-finger collision pairs in {urdf} (fingers {sorted(set(finger))})"

    def _shift_depth(self, a: int, b: int) -> float:
        """Penetration of a pair whose EPA failed: distance b must be moved away from a (along the line between the
        geometry centres) until the boolean GJK test reports no collision; bisection to 0.05 mm, capped at 5 cm."""
        ga, gb = self.geom.geometryObjects[a].geometry, self.geom.geometryObjects[b].geometry
        Ma, Mb = self.gdata.oMg[a], self.gdata.oMg[b]
        n = Mb.translation - Ma.translation
        n = n / max(np.linalg.norm(n), 1e-9)
        Ta = fcl.Transform3f(Ma.rotation, Ma.translation)
        req = fcl.CollisionRequest()

        def hit(t):
            res = fcl.CollisionResult()
            return fcl.collide(ga, Ta, gb, fcl.Transform3f(Mb.rotation, Mb.translation + t * n), req, res) > 0

        lo, hi = 0.0, 0.05
        if hit(hi):
            return 0.05
        while hi - lo > 5e-5:
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if hit(m) else (lo, m)
        return hi

    def penetration(self, dex_q: np.ndarray) -> tuple[float, tuple[str, str] | None]:
        """Deepest finger-finger penetration (m, 0 = none) and its geometry pair. Collision test with contacts (GJK/EPA),
        much faster than the full distance query.

        EPA fails (inf / nan depth) on nearly touching hulls, depending on the query (the same pose measured 1.5 mm or
        inf, 2026-11-11); a failed pair is measured by _shift_depth instead (separating translation, an upper bound of the
        penetration depth)."""
        q = pin.neutral(self.model)
        q[self.q_index] = dex_q
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, q)
        pin.computeCollisions(self.geom, self.gdata, False)
        best, pair = 0.0, None
        for i, res in enumerate(self.gdata.collisionResults):
            if self.active[i] and res.isCollision():
                a, b = self.pairs[i]
                ds = [abs(res.getContact(k).penetration_depth) for k in range(res.numContacts())]
                ok = ds and all(np.isfinite(x) and x < 0.05 for x in ds)   # EPA failure: inf / nan / ~1e308
                d = max(ds) if ok else self._shift_depth(a, b)
                if d > best:
                    best, pair = d, (self.names[a], self.names[b])
        return best, pair

    def enable_contacts(self, dex_q_open: np.ndarray | None = None):
        for req in self.gdata.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = 1
        # pairs already in contact at the open pose (coarse hulls of neighbouring links) are not finger crossings
        q0 = np.zeros(len(self.q_index)) if dex_q_open is None else dex_q_open
        q = pin.neutral(self.model)
        q[self.q_index] = q0
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, q)
        pin.computeCollisions(self.geom, self.gdata, False)
        base = [i for i, r in enumerate(self.gdata.collisionResults) if r.isCollision()]
        for i in base:
            self.geom.collisionPairs[i]   # keep index valid for names
        self.active = np.ones(len(self.pairs), bool)
        self.active[base] = False
        for i in base:
            self.gdata.deactivateCollisionPair(i)
        self.n_base_excluded = len(base)


_COL: dict = {}


def _init(rt, hands, mesh_roots):
    meta = json.load(open(os.path.join(rt, "meta.json")))
    index = _mesh_index(mesh_roots.split(","))
    tmp = tempfile.mkdtemp()
    for h in hands:
        c = HandCollision(resolved_urdf(h, os.path.join(rt, "urdf", f"{h}.urdf"), meta["hands"][h]["urdf_orig"], index, tmp),
                          meta["hands"][h]["dex_joint_names"])
        c.enable_contacts()
        _COL[h] = c


def _check(job):
    f, hands, stride, pen_mm = job
    z = np.load(f)
    r = {"label": str(z["action_label"]), "T": int(len(z["t"]))}
    for h in hands:
        idx = np.arange(0, len(z["t"]), stride)
        res = [_COL[h].penetration(z[f"{h}_dexq"][k]) for k in idx]
        d = np.array([x[0] for x in res])
        w = int(np.argmax(d))
        r[h] = {"pen_frac": round(float((d > pen_mm / 1000).mean()), 3), "max_pen_mm": round(float(d.max() * 1000), 1),
                "worst_frame": int(idx[w]), "worst_pair": res[w][1],
                "pen_frames": idx[d > pen_mm / 1000].tolist()}
    return os.path.basename(f)[:-4], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--traj", default="", help="comma list of trajectory ids (default: all)")
    ap.add_argument("--hands", default="")
    ap.add_argument("--pen_mm", type=float, default=2.0, help="penetration deeper than this counts as infeasible")
    ap.add_argument("--mesh_roots", default="/workspace/assets/dexverse,/workspace/assets/bench2dex")
    ap.add_argument("--stride", type=int, default=4, help="check every n-th frame (60 Hz -> 15 Hz)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.rt, "meta.json")))
    hands = a.hands.split(",") if a.hands else list(meta["hands"])
    files = sorted(glob.glob(os.path.join(a.rt, "*.npz")))
    if a.traj:
        files = [os.path.join(a.rt, f"{t}.npz") for t in a.traj.split(",")]
    rep = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.rt, hands, a.mesh_roots)) as ex:
        for tid, r in ex.map(_check, [(f, hands, a.stride, a.pen_mm) for f in files]):
            rep[tid] = r
            print(tid, r["label"], {h: (r[h]["pen_frac"], r[h]["max_pen_mm"]) for h in hands}, flush=True)
    summ = {h: {"pen_frac_mean": round(float(np.mean([r[h]["pen_frac"] for r in rep.values()])), 3),
                "traj_with_pen": int(sum(r[h]["pen_frac"] > 0 for r in rep.values())), "n": len(rep)} for h in hands}
    print("summary", summ)
    if a.out:
        json.dump({"per_traj": rep, "summary": summ, "args": vars(a)}, open(a.out, "w"), indent=1)
        save_manifest(a.out, a, summary=summ)


if __name__ == "__main__":
    main()
