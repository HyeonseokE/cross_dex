"""FK sanity of the pilot hand models: joint names / limits from the URDF, tip positions and finger lengths (mm)."""
import os
import sys

import torch

from contact_match import SPEC
from urdf_fk import Hand

for h, s in SPEC.items():
    H = Hand(os.path.join(sys.argv[1], s[0]), s[1], s[2])
    z = torch.zeros(1, len(H.names), dtype=torch.double); m = ((H.lower + H.upper) / 2)[None]
    cz, cm = H.chain_points(z), H.chain_points(m); ds = H.distal_start()
    print(h, "dof", len(H.names), "mimic", sum(j["mimic"] is not None for j in H.joints.values()), H.names[:6])
    for f, c in enumerate(cz):
        L = (c[0, 1:] - c[0, :-1]).norm(dim=-1).sum() * 1e3
        print("  f%d links %d len %.0f distal %.0f tip0 %s tipmid %s" % (f, c.shape[1], L, (c[0, -1] - c[0, ds[f]]).norm() * 1e3,
              [round(float(v) * 1e3) for v in c[0, -1]], [round(float(v) * 1e3) for v in cm[f][0, -1]]))
