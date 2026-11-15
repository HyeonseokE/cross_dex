"""Left LEAP joint targets for the DexJoCo-X left-hand LAD pairs (2026-11-15).

DexJoCo-X's left LEAP is not a mirror of its right LEAP: the three fingers are laid out the same way, but the thumb is
mounted elsewhere on the palm (thumb base to the finger bases: 62.5 / 79.5 / 113.5 mm left vs 66.9 / 65.9 / 91.2 mm right),
so no joint sign map and no rigid mirror of the palm frame carries right poses to left poses (16 mm in the MuJoCo
left/right test; the best sign map leaves 6 mm mean / 33 mm max at the fingertips). The left pairs of this hand therefore
cannot be made by the joint-wise conversion used for the other hands (build_pairs_bench.left_map).

Instead every right-LEAP pose of the pair file is retargeted onto the left LEAP model by its hand-internal geometry, which
does not depend on any frame or on the mirror: the distances between the keypoints (4 fingertips, 8 finger joint
positions, 3 finger bases) of the left hand are fitted to those of the right hand in that pose (key-vector lengths, as in
DexPilot / the LAD paper's retargeting), by batched gradient descent on the differentiable FK of the URDFs written from the
MuJoCo models. Start = the best sign map, joint limits enforced. Row i of the result still belongs to the same human pose
as row i of every other hand.

  /workspace/envs/dexverse/dexlatent/bin/python -m dexverse_lad.leap_left_retarget \
      --pairs <dexjoco_right_pairs.npy> --out <dexjoco_left_leap.npy>
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("DEXLATENT_ROOT", "/workspace/codes/dexverse/DexLatent"))
import HandLatent.kinematics as K  # noqa: E402

URDF = os.path.join(os.environ.get("DEXLATENT_ROOT", "/workspace/codes/dexverse/DexLatent"), "dexverse_xlvla", "dexjoco_urdf")
TIPS = ["thumb_fingertip_tip", "fingertip_tip", "fingertip_2_tip", "fingertip_3_tip"]
MIDS = ["thumb_fingertip", "thumb_dip", "fingertip", "dip", "fingertip_2", "dip_2", "fingertip_3", "dip_3"]
BASE = ["mcp_joint", "mcp_joint_2", "mcp_joint_3"]                   # the three finger bases (same layout on both hands)
NT, NK = len(TIPS), len(TIPS) + len(MIDS) + len(BASE)
_I, _J = torch.triu_indices(NK, NK, 1).unbind()
_keep = ~((_I >= NK - len(BASE)) & (_J >= NK - len(BASE)))            # base-base distances are constants
PI, PJ = _I[_keep], _J[_keep]
PW = torch.where((PI < NT) | (PJ < NT), 1.0, 0.3)                     # pairs with a fingertip matter most
TT = (PI < NT) & (PJ < NT)                                            # fingertip-fingertip pairs (reported)


def model(side):
    return K.HandKinematicsModel(hand_name=f"leap_{side}", urdf_path=f"{URDF}/leap_{side}.urdf", root_link="hand_root",
                                 tip_links=TIPS + MIDS + BASE)


def fk(m, ang):
    return m._forward_internal(m.angles_to_normalized(ang))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=400000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (test)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R, L = model("right"), model("left")
    perm = {s: [int(j) for j in m.dof_joints] for s, m in (("right", R), ("left", L))}     # model dof -> actuator index
    inv = {s: np.argsort(p) for s, p in perm.items()}
    q = np.load(a.pairs, allow_pickle=True).item()["leap"]
    if a.limit:
        q = q[:a.limit]
    pi, pj, pw, tt = PI.to(dev), PJ.to(dev), PW.to(dev), TT.to(dev)
    dist = lambda P: (P[:, pi] - P[:, pj]).norm(dim=-1)              # [B, pairs]
    loL, hiL = L._lower.to(dev), L._upper.to(dev)
    sgn = torch.ones(16, device=dev); sgn[[4, 13]] = -1               # start: best sign map (actuator order)
    out = np.empty_like(q)
    tip_err, init_err, t0 = [], [], time.time()
    for s in range(0, len(q), a.batch):
        qa = torch.from_numpy(q[s:s + a.batch]).to(dev)               # right, actuator order
        with torch.no_grad():
            Y = dist(fk(R, qa[:, perm["right"]]))
        x = (qa * sgn)[:, perm["left"]].clamp(loL, hiL).clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=a.lr)
        for _ in range(a.iters):
            opt.zero_grad()
            loss = (((dist(fk(L, x)) - Y) ** 2) * pw).sum(1).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(loL, hiL)
        with torch.no_grad():
            e = (dist(fk(L, x)) - Y).abs()[:, tt]                      # fingertip-fingertip distance error per row / pair
            tip_err.append(e.cpu())
            e0 = (dist(fk(L, (qa * sgn)[:, perm["left"]].clamp(loL, hiL))) - Y).abs()[:, tt]
            init_err.append(e0.mean().item())
            out[s:s + a.batch] = x[:, torch.as_tensor(inv["left"], device=dev)].cpu().numpy()
        print(f"[leap-left] rows {s + len(qa)}/{len(q)} tip-tip distance err mean {e.mean() * 1e3:.2f} mm (sign map {e0.mean() * 1e3:.2f}) ({time.time() - t0:.0f}s)", flush=True)
    e = torch.cat(tip_err).numpy() * 1e3
    np.save(a.out, out)
    stats = {"rows": int(len(q)), "sign_map_tip_tip_err_mean_mm": round(float(np.mean(init_err)) * 1e3, 3), "tip_tip_dist_err_mm": {"mean": round(float(e.mean()), 3), "p95": round(float(np.percentile(e, 95)), 3),
                                               "p99": round(float(np.percentile(e, 99)), 3), "max": round(float(e.max()), 3)},
             "iters": a.iters, "lr": a.lr, "keypoints": TIPS + MIDS + BASE, "pair_weights": "1.0 with a fingertip, 0.3 otherwise",
             "method": "keypoint distances of the left hand fitted to the right hand's (frame free), gradient descent on the FK",
             "pairs": a.pairs, "order": "DexJoCo-X actuator order (leap_0 .. leap_15)"}
    json.dump(stats, open(a.out[:-4] + ".meta.json", "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ("sign_map_tip_tip_err_mean_mm", "tip_tip_dist_err_mm")}), flush=True)


if __name__ == "__main__":
    main()
