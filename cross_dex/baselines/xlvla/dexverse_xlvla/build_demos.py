#!/usr/bin/env python
"""Write the cross-hand replay pickle with the SIMULATOR env's python (numpy 1.x): a pickle written by the baseline
env (numpy 2) cannot be loaded by the evaluator. Only pickle / numpy / json; run by file path, e.g.
  /workspace/envs/dexverse/main/bin/python build_demos.py <spec.json>
spec: {"src_pkl", "dst_pkl", "out", "src_hand", "mode", "episodes": [...],
       "start": {"<k>": {"<sim joint name>": value, ...}}}   B's starting joint values (wrist + fingers, incl. mimic)
Per episode k of B's pickle: objects (rigid_object) and goal from A's episode k, B's robot start joints from spec.
"""
import copy
import json
import pickle
import sys

import numpy as np

spec = json.load(open(sys.argv[1]))
src = pickle.load(open(spec["src_pkl"], "rb"))
dst = pickle.load(open(spec["dst_pkl"], "rb"))
sim_names = dst["action_layout"]["robot_joint_names"]
for k in spec["episodes"]:
    eb, ea = dst["episodes"][k], src["episodes"][k]
    st = copy.deepcopy(eb["initial_state"])
    st["rigid_object"] = copy.deepcopy(ea["initial_state"]["rigid_object"])
    jp = np.array(st["articulation"]["robot"]["joint_position"], dtype=np.float32)
    for n, v in spec["start"][str(k)].items():
        if n in sim_names:
            jp[0, sim_names.index(n)] = v
    st["articulation"]["robot"]["joint_position"] = jp
    st["articulation"]["robot"]["joint_velocity"] = np.zeros_like(jp)
    eb["initial_state"] = st
    eb["goal_pose"] = ea.get("goal_pose")
    eb["source"] = {"hand": spec["src_hand"], "episode": k}
dst["xlvla_replay"] = {"src": spec["src_hand"], "src_pkl": spec["src_pkl"], "mode": spec["mode"]}
pickle.dump(dst, open(spec["out"], "wb"))
print(spec["out"])
