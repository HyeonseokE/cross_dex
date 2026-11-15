"""DexVerse hands for the XL-VLA latent autoencoder (DexLatent).

The official code mounts four hands on one xArm7, so their fingertips share one world frame and the arm joints pass
through the latent unchanged. DexVerse hands are floating (six virtual wrist joints) and every hand URDF uses its own
palm frame. This module therefore

* builds a hand-only URDF (joints reachable from the palm root; Shadow's six floating joints are dropped),
* expresses fingertips in a hand-common CANONICAL frame measured from the open hand
  (x: wrist -> mean non-thumb fingertip, y: toward the thumb side, z: x cross y; origin: the pinch point),
* maps between the LeRobot dataset joint order (sim names) and the URDF DoF order.

The hand latent covers the finger joints only (``arm_dof = 0``); the wrist is handled outside the latent with the
hand-common wrist pose recorded in the dataset (``action.ee_abs_rot6d``).

Fingertip order is thumb, index, middle, ring, little (DexLatent's pinch pairs are (0, k)). Allegro and LEAP have no
little finger, so pairs with index 4 drop out for them (DexLatent ``shared_pinch_pairs``), as the paper does for
Paxini DexH13.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as et
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

import HandLatent.kinematics as K

ASSET_ROOT = os.environ.get("DEXVERSE_ROBOT_AGENTS", "/workspace/assets/dexverse/robot_agents")
# hand models DexVerse's retarget folders lack (2026-11-15: Inspire left = dex-urdf robots/hands/inspire_hand/inspire_hand_left.urdf,
# the mirror of DexVerse's right Inspire URDF: same joint names, limits and mimic ratios)
EXTRA_URDF_ROOT = os.environ.get("XLVLA_EXTRA_URDF", "/workspace/assets/dexverse/xlvla_extra")
URDF_CACHE = os.environ.get("XLVLA_URDF_CACHE", "/workspace/.cache/xlvla_dexverse_urdf")


@dataclass(frozen=True)
class HandSpec:
    urdf: str                    # relative to ASSET_ROOT
    root_link: str               # palm root (FK origin before canonicalisation)
    tip_links: Sequence[str]     # thumb, index, middle, ring[, little]
    data_name: Dict[str, str]    # dataset joint name -> URDF joint name, only where they differ


HANDS: Dict[str, HandSpec] = {
    "shadow": HandSpec(
        "shadow/retarget/floating_shadow_right.urdf", "",   # root = child of the last floating joint (_shadow_root)
        ("thtip", "fftip", "mftip", "rftip", "lftip"), {}),
    "allegro": HandSpec(
        "allegro/retarget/allegro_hand_right/allegro_hand_right.urdf", "base_link",
        ("link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"),
        {f"joint_{i}_0": f"joint_{i}.0" for i in range(16)}),
    "inspire": HandSpec(
        "inspire/retarget/inspire_hand_right/inspire_hand_right.urdf", "base",
        ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"), {}),
    "leap": HandSpec(
        "leap/retarget/leap_hand_right_fixed.urdf", "base_link",
        ("thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"), {}),
    "sharpa": HandSpec(
        "sharpa/retarget/sharpa_wave_right/sharpa_wave_right.urdf", "right_hand_C_MC",
        ("right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip",
         "right_pinky_fingertip"), {}),
    "wuji": HandSpec(
        "wuji/retarget/wuji_hand_right/wuji_hand_right.urdf", "right_palm_link",
        ("right_finger1_tip_link", "right_finger2_tip_link", "right_finger3_tip_link", "right_finger4_tip_link",
         "right_finger5_tip_link"), {}),
}
HAND_NAMES: List[str] = list(HANDS)            # the six right hands (defaults everywhere stay right-handed)

_SHADOW_J = [f"{f}J{i}" for f, n in (("FF", 4), ("MF", 4), ("RF", 4), ("LF", 5), ("TH", 5)) for i in range(1, n + 1)]
# left hands for bimanual tasks (2026-11-15): dataset joint names of the bimanual LeRobot sets (left_* / lh_*) -> URDF names
LEFT_HANDS: Dict[str, HandSpec] = {
    "shadow_left": HandSpec(
        "shadow/retarget/floating_shadow_left.urdf", "",
        ("thtip", "fftip", "mftip", "rftip", "lftip"), {f"lh_{j}": j for j in _SHADOW_J}),
    # left Allegro URDF: the joint_0-3 finger is the one FARTHEST from the thumb (open hand: 329 mm vs 261 mm on the
    # right) -> fingertip order thumb, then outward from the thumb = link_11, link_7, link_3 (checked 2026-11-15)
    "allegro_left": HandSpec(
        "allegro/retarget/allegro_hand_left/allegro_hand_left.urdf", "base_link",
        ("link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip"),
        {f"left_joint_{i}_0": f"joint_{i}.0" for i in range(16)}),
    "inspire_left": HandSpec(
        os.path.join(EXTRA_URDF_ROOT, "inspire_hand_left.urdf"), "base",
        ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"),
        {f"left_{j}": j for j in ("index_proximal_joint", "middle_proximal_joint", "pinky_proximal_joint",
                                  "ring_proximal_joint", "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint")}),
    "leap_left": HandSpec(
        "leap/retarget/leap_hand_left_fixed.urdf", "base_link",
        ("thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"), {f"left_{i}": f"leap_{i}" for i in range(16)}),
    "sharpa_left": HandSpec(
        "sharpa/retarget/sharpa_wave_left/sharpa_wave_left.urdf", "left_hand_C_MC",
        ("left_thumb_fingertip", "left_index_fingertip", "left_middle_fingertip", "left_ring_fingertip",
         "left_pinky_fingertip"), {}),
    "wuji_left": HandSpec(
        "wuji/retarget/wuji_hand_left/wuji_hand_left.urdf", "left_palm_link",
        ("left_finger1_tip_link", "left_finger2_tip_link", "left_finger3_tip_link", "left_finger4_tip_link",
         "left_finger5_tip_link"), {}),
}
LEFT_HAND_NAMES: List[str] = list(LEFT_HANDS)
HANDS.update(LEFT_HANDS)

# DexJoCo-X hands that DexVerse lacks (dj_wuji, dj_linkerhand, dj_xhand, right and left); see hands_dexjoco.py
from dexverse_xlvla import hands_dexjoco as _DJ   # noqa: E402

HANDS.update(_DJ.specs(HandSpec, URDF_CACHE, ASSET_ROOT))
SHADOW_FLOATING_CHILD_OF = "z_rotation_joint"   # Shadow hand root = child link of the last floating joint


def _shadow_root(root: et.Element) -> str:
    for j in root.findall("joint"):
        if j.get("name") == SHADOW_FLOATING_CHILD_OF:
            return j.find("child").get("link")
    raise RuntimeError("Shadow URDF: floating joint not found")


def hand_only_urdf(hand: str) -> tuple[str, str]:
    """Write the hand-only URDF (joints reachable from the palm root) and return (path, root_link)."""
    spec = HANDS[hand]
    tree = et.parse(os.path.join(ASSET_ROOT, spec.urdf))
    root = tree.getroot()
    root_link = _shadow_root(root) if hand.startswith("shadow") else spec.root_link
    if root_link not in {l.get("name") for l in root.findall("link")}:
        raise RuntimeError(f"{hand}: root link {root_link!r} not in {spec.urdf}")
    children: Dict[str, List[et.Element]] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    keep_links, keep_joints, queue = {root_link}, set(), [root_link]
    while queue:
        link = queue.pop()
        for j in children.get(link, []):
            keep_joints.add(id(j))
            child = j.find("child").get("link")
            keep_links.add(child)
            queue.append(child)
    for el in list(root):
        if (el.tag == "joint" and id(el) not in keep_joints) or (el.tag == "link" and el.get("name") not in keep_links):
            root.remove(el)
    os.makedirs(URDF_CACHE, exist_ok=True)
    path = os.path.join(URDF_CACHE, f"{hand}_hand.urdf" if hand.endswith("_left") else f"{hand}_right_hand.urdf")
    tree.write(path)
    return path, root_link


class CanonicalHandModel(K.HandKinematicsModel):
    """DexLatent FK model whose fingertips and wrist pose are expressed in the hand-common canonical frame."""

    def __init__(self, hand: str):
        path, root_link = hand_only_urdf(hand)
        super().__init__(hand_name=hand, urdf_path=path, root_link=root_link, tip_links=HANDS[hand].tip_links)
        missing = [t for t in self.tip_links if t not in {j.child for j in self.joint_specs.values()}]
        if missing:
            raise RuntimeError(f"{hand}: tip links not in URDF: {missing}")
        self.canon = torch.eye(4)
        self.canon = self._measure_canonical_frame()

    # --- canonical frame -------------------------------------------------------------------------------------
    def _raw(self, normalized: torch.Tensor):
        tips, wrist, sq = super()._forward_internal(normalized)
        return tips, wrist, sq

    def open_qpos(self) -> torch.Tensor:
        """Joint angle 0 clipped to the limits (the URDF zero pose is the open hand for all six hands)."""
        return self.angles_to_normalized(torch.zeros(self.dof_count())).clamp(-1, 1)

    def _measure_canonical_frame(self) -> torch.Tensor:
        tips, _, _ = self._raw(self.open_qpos().unsqueeze(0))
        tips = tips[0]
        thumb, fingers = tips[0], tips[1:]
        x = fingers.mean(0)                                   # root -> mean fingertip (root at origin)
        x = x / x.norm()
        side = fingers[0] - fingers[-1]                       # last finger -> index: toward the thumb side
        y = side - (side @ x) * x
        y = y / y.norm()
        z = torch.linalg.cross(x, y)
        rot = torch.stack([x, y, z], 1)                       # columns = canonical axes in the root frame
        # pinch point (DexLatent's pinch templates bring the thumb and one finger to it): half the fingertip reach
        # along x, a quarter of the reach toward the palm side. Palm side = where the fingertips go when the
        # joints move to mid-range (fingers curl toward the palm). Scaled by the reach so it fits every hand size.
        reach = fingers.mean(0) @ x
        mid, _, _ = self._raw(torch.zeros(1, self.dof_count()))
        palm_dir = torch.sign(((mid[0, 1:] - fingers) @ z).mean())
        origin = 0.5 * reach * x + 0.25 * reach * palm_dir * z
        t = torch.eye(4)
        t[:3, :3] = rot.T
        t[:3, 3] = -(rot.T @ origin)
        return t                                               # root frame -> canonical frame

    def _forward_internal(self, normalized_qpos: torch.Tensor):
        tips, wrist, sq = self._raw(normalized_qpos)
        c = self.canon.to(device=tips.device, dtype=tips.dtype)
        tips = tips @ c[:3, :3].T + c[:3, 3]
        wrist = c.unsqueeze(0) @ wrist
        return tips, wrist, sq

    # --- dataset <-> model joint order -----------------------------------------------------------------------
    def data_index(self, data_names: Sequence[str]) -> List[int]:
        """Column of each model DoF in a dataset vector with names ``data_names`` (sim joint names)."""
        alias = HANDS[self.hand_name].data_name
        urdf_of = [alias.get(n, n) for n in data_names]
        missing = [j for j in self.dof_joints if j not in urdf_of]
        if missing:
            raise KeyError(f"{self.hand_name}: dataset lacks joints {missing}")
        return [urdf_of.index(j) for j in self.dof_joints]


class DexVerseFK(K.MultiHandDifferentiableFK):
    """Registry with the same interface as DexLatent's ``MultiHandDifferentiableFK``."""

    def __init__(self, hand_names=None):
        self.models = {h: CanonicalHandModel(h) for h in (hand_names or HAND_NAMES)}


def install() -> None:
    """Make DexLatent's trainer build DexVerse hands (call before constructing ``CrossEmbodimentTrainer``)."""
    import HandLatent.model as M
    M.MultiHandDifferentiableFK = DexVerseFK
