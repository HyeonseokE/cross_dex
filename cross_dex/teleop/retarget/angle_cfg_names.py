"""Bench2Dex hand_type -> retarget/angle_cfgs file name (the angle map and the DexPilot prior use the same file)."""

_ANGLE_CFG_NAME = {"rh56dfx": "inspire"}


def angle_cfg_name(hand_type: str) -> str | None:
    return _ANGLE_CFG_NAME.get(hand_type, hand_type)
