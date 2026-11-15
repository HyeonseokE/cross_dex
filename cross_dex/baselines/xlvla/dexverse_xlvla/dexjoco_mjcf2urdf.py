"""URDF of one hand of a DexJoCo-X MuJoCo model, written from the compiled model itself (exact kinematics: body offsets,
hinge axes, ranges, actuator order), for hands whose shipped URDF does not match the MuJoCo model (left LEAP).
Tip links = the fingertip geoms used in the checks (farthest geom of each leaf body from the hand root)."""
import json, sys, numpy as np, mujoco
DJ = "/workspace/codes/dexjoco/DexJoCo-X"
h, side, out = sys.argv[1], sys.argv[2], sys.argv[3]
task = "bimanual_photograph"
m = mujoco.MjModel.from_xml_path(f"{DJ}/simulator/dexjoco/sim/envs/xmls/benchmark7_v3_{task}_{h}.xml"); d = mujoco.MjData(m)
con = json.load(open(f"{DJ}/evaluation/contracts/{task}/{h}.json"))
if m.nkey: mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d); par = m.body_parentid
def anc(b):
    o = [b]
    while b: b = int(par[b]); o.append(b)
    return o
pre = f"benchhand_{side}_native_"
short = lambda n: n.replace(pre, "").replace("/", "_")
hj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in con["joint_names"][side]]
hb = {int(m.jnt_bodyid[j]) for j in hj}; common = set(anc(next(iter(hb))))
for b in hb: common &= set(anc(b))
root = max(common, key=lambda b: len(anc(b))); sub = [b for b in range(m.nbody) if root in anc(b) and b != root]
leaves = [b for b in sub if b not in {int(par[x]) for x in sub}]
def R(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)
def rpy(M):
    if abs(M[2, 0]) > 1 - 1e-9:                      # pitch = +-90 deg (gimbal lock): yaw = 0
        return (np.arctan2(M[0, 1], M[1, 1]), np.pi / 2, 0.0) if M[2, 0] < 0 else (np.arctan2(-M[0, 1], M[1, 1]), -np.pi / 2, 0.0)
    return np.arctan2(M[2, 1], M[2, 2]), np.arctan2(-M[2, 0], np.hypot(M[2, 1], M[2, 2])), np.arctan2(M[1, 0], M[0, 0])
jof = {}
for b in sub:
    js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    assert len(js) <= 1 and all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in js), "one hinge per body expected"
    jof[b] = js[0] if js else None
# joint equality constraints (q1 = c0 + c1 * q2: coupled distal joints of Inspire / LinkerHand) -> URDF mimic
mimic = {}
for e in range(m.neq):
    if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]:
        c = m.eq_data[e][:5]
        assert np.abs(c[2:]).max() < 1e-9, "non-linear joint coupling"
        mimic[int(m.eq_obj1id[e])] = (int(m.eq_obj2id[e]), float(c[1]), float(c[0]))
jp = lambda b: m.jnt_pos[jof[b]] if (b in jof and jof[b] is not None) else np.zeros(3)
L = ['<?xml version="1.0"?>', f'<robot name="dj_{h}_{side}">', '  <link name="hand_root"/>']
for b in sub:
    p = int(par[b]); pn = "hand_root" if p == root else short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, p)); n = short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
    Rb = R(m.body_quat[b]); xyz = m.body_pos[b] + Rb @ jp(b) - (jp(p) if p != root else 0); r, pch, y = rpy(Rb)
    L.append(f'  <link name="{n}"/>')
    if jof[b] is None:
        L.append(f'  <joint name="{n}_fixed" type="fixed"><origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" rpy="{r:.9f} {pch:.9f} {y:.9f}"/><parent link="{pn}"/><child link="{n}"/></joint>')
    else:
        j = jof[b]; ax = m.jnt_axis[j]; lo, hi = m.jnt_range[j]
        L.append(f'  <joint name="{short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))}" type="revolute"><origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" rpy="{r:.9f} {pch:.9f} {y:.9f}"/>'
                 f'<parent link="{pn}"/><child link="{n}"/><axis xyz="{ax[0]:.9f} {ax[1]:.9f} {ax[2]:.9f}"/><limit lower="{lo:.6f}" upper="{hi:.6f}" effort="10" velocity="10"/>'
                 + (f'<mimic joint="{short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, mimic[j][0]))}" multiplier="{mimic[j][1]:.6f}" offset="{mimic[j][2]:.6f}"/>' if j in mimic else '') + '</joint>')
tips = []
for b in leaves:
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (b, int(par[b]))]
    g = max(gs, key=lambda g: np.linalg.norm(d.geom_xpos[g] - d.xpos[root])); n = short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
    gb = int(m.geom_bodyid[g]); gn = short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, gb))     # the geom may sit on the leaf's parent body
    off = d.xmat[gb].reshape(3, 3).T @ (d.geom_xpos[g] - d.xpos[gb]) - jp(gb)
    L.append(f'  <link name="{n}_tip"/>'); L.append(f'  <joint name="{n}_tip_fixed" type="fixed"><origin xyz="{off[0]:.9f} {off[1]:.9f} {off[2]:.9f}" rpy="0 0 0"/><parent link="{gn}"/><child link="{n}_tip"/></joint>')
    tips.append((n + "_tip", d.geom_xpos[g].copy()))
L.append("</robot>"); open(out, "w").write("\n".join(L))
acts = [short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), 0])) for a in con["actuator_names"][side]]
json.dump({"hand": h, "side": side, "actuator_joints": acts, "tips": [t for t, _ in tips], "mimic_joints": len(mimic),
           "source": f"benchmark7_v3_{task}_{h}.xml"}, open(out[:-5] + ".json", "w"), indent=1)
print("wrote", out, "| actuator-order joints:", acts)
print("tips:", [t for t, _ in tips]); th = ([x for t, x in tips if "thumb" in t or t.startswith("th_")] or [tips[0][1]])[0]
print("thumb-to-finger distance at home (mm):", {t: round(float(np.linalg.norm(x - th)) * 1e3, 1) for t, x in tips})
