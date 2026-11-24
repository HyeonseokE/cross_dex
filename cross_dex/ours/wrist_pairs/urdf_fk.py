"""Minimal differentiable URDF forward kinematics (torch) for the contact-matching pilot (RSS2027, 2026-11-23).

Only what the pilot needs: revolute / continuous / prismatic / fixed joints, mimic joints, FK of the subtree below a
root link, finger chains root -> tip. Joint order = order of the non-mimic movable joints in the URDF (names in
``self.names``; limits from the URDF).
"""
import xml.etree.ElementTree as ET

import torch


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (torch.cos(torch.tensor(r)), torch.sin(torch.tensor(r)), torch.cos(torch.tensor(p)),
                              torch.sin(torch.tensor(p)), torch.cos(torch.tensor(y)), torch.sin(torch.tensor(y)))
    return torch.stack([torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
                        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
                        torch.stack([-sp, cp * sr, cp * cr])]).double()


def axis_rot(axis, ang):   # axis [3], ang [B] -> [B,3,3] (Rodrigues)
    K = torch.zeros(3, 3, dtype=ang.dtype, device=ang.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, dtype=ang.dtype, device=ang.device) + s * K + (1 - c) * (K @ K)


class Hand:
    def __init__(self, urdf, root_link, tip_links):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin"); ax = j.find("axis"); lim = j.find("limit"); mim = j.find("mimic")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            ang = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = torch.eye(4, dtype=torch.double); T[:3, :3] = rpy(*ang); T[:3, 3] = torch.tensor(xyz, dtype=torch.double)
            a = torch.tensor([float(v) for v in ax.get("xyz").split()], dtype=torch.double) if ax is not None else torch.tensor([1., 0, 0], dtype=torch.double)
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "T": T, "axis": a / a.norm(),
                "lim": (float(lim.get("lower", -3.14)), float(lim.get("upper", 3.14))) if lim is not None else (-3.14, 3.14),
                "mimic": (mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0))) if mim is not None else None}
        self.root = root_link
        self.chains = []                                    # per tip: links from the root (excluded) to the tip
        for t in tip_links:
            c = [t]
            while c[0] != root_link:
                c.insert(0, self.joints[c[0]]["parent"])
            self.chains.append(c[1:])
        links = []
        for c in self.chains:                               # root -> tip order keeps parents before children
            links += [l for l in c if l not in links]
        self.links = links
        mov = [self.joints[l] for l in links if self.joints[l]["type"] != "fixed" and self.joints[l]["mimic"] is None]
        self.names = [j["name"] for j in mov]
        self.lower = torch.tensor([j["lim"][0] for j in mov], dtype=torch.double); self.upper = torch.tensor([j["lim"][1] for j in mov], dtype=torch.double)
        self.idx = {n: i for i, n in enumerate(self.names)}

    def fk(self, q):   # q [B, n] -> {link: [B,4,4]} in the root-link frame
        B = q.shape[0]; out = {self.root: torch.eye(4, dtype=q.dtype, device=q.device).expand(B, 4, 4)}
        for l in self.links:
            j = self.joints[l]; T = out[j["parent"]] @ j["T"].to(q)
            if j["type"] != "fixed":
                v = q[:, self.idx[j["name"]]] if j["mimic"] is None else q[:, self.idx[j["mimic"][0]]] * j["mimic"][1] + j["mimic"][2]
                M = torch.eye(4, dtype=q.dtype, device=q.device).repeat(B, 1, 1)
                if j["type"] == "prismatic":
                    M[:, :3, 3] = v[:, None] * j["axis"].to(q)
                else:
                    M[:, :3, :3] = axis_rot(j["axis"].to(q), v)
                T = T @ M
            out[l] = T
        return out

    def chain_points(self, q):   # per finger: [B, n_links, 3] link origins from the first movable joint's child to the tip
        P = self.fk(q); res = []
        for c in self.chains:
            first = next(i for i, l in enumerate(c) if self.joints[l]["type"] != "fixed")
            res.append(torch.stack([P[l][:, :3, 3] for l in c[first:]], 1))
        return res

    def distal_start(self):      # per finger: index (within chain_points) where the distal segment starts
        res = []
        for c in self.chains:
            first = next(i for i, l in enumerate(c) if self.joints[l]["type"] != "fixed")
            last = max(i for i, l in enumerate(c) if self.joints[l]["type"] != "fixed")
            res.append(min(last - first, len(c) - first - 2))
        return res
