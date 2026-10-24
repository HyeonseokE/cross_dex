# teleop/retarget/ — 손가락 리타게팅 (시뮬레이터 무관)

| 경로 | 내용 |
|---|---|
| `bench2dex_finger_retargeter.py` | `Bench2DexHand`: OpenXR 손 관절 26개 → MediaPipe 21점 + Bench2Dex 손목 좌표 → 로봇 관절 각도. OpenXR→Bench2Dex 축 변환 `(+Y, −X, +Z)` |
| `bench2dex/` | Bench2Dex 텔레옵 코드(`bench2dex_teleop/teleop`, 커밋 511cbf3)의 `retarget_bridge.py`와 `hand_cfgs/`를 **수정 없이** 복사 |

- URDF는 Bench2Dex URDF-Zoo(`$URDF_ZOO_DIR`).
- 손 보정값: `bench2dex/hand_cfgs/<손>.yml`의 `hand_scale_xyz`, `hand_offset_xyz`(Manus 기준), `joint_limits_override`.
  실행 중에 고치면 약 2초 안에 반영된다. 지원 손: shadow, allegro, leap, sharpa, wuji, rh56dfx(Inspire) 등 12종.
- 출력 관절은 이름으로 로봇 action에 맞춘다(`dexverse_integration.py`).
