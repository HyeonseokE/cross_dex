import os, pickle, sys, time
import numpy as np, torch
torch.set_num_threads(1)
from contact_match import SPEC, Hand, pieces, planes, solve, sdf_true
G = pickle.load(open("pilot_grasps.pkl", "rb")); g = G[0]
t = time.time(); ps = pieces(g); print("pieces", len(ps), sum(len(m.vertices) for m in ps), round(time.time() - t, 2), flush=True)
t = time.time(); P = planes(ps); print("planes", tuple(P.shape), round(time.time() - t, 2), flush=True)
for h in ("inspire", "sharpa"):
    H = Hand(os.path.join("urdf", SPEC[h][0]), SPEC[h][1], SPEC[h][2])
    t = time.time(); s = solve(H, SPEC[h][3], g, P, 20); print(h, "20 iters", round(time.time() - t, 2), s["pts"].shape, flush=True)
    t = time.time(); d = sdf_true(ps, s["pts"][0]); print("sdf_true", round(time.time() - t, 2), flush=True)
