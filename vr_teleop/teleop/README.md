# teleop/ — 기기 조합 → 로봇 (텔레옵 실행 중 코드)

| 파일 | 의존 | 내용 |
|---|---|---|
| `hand_mounted_controller.py` | numpy | 손등에 단 컨트롤러 pose → 손목 6D pose (`t + R·offset`, `R·mount_rot`). 값은 `tools/calibration` 결과 |
| `retarget/` | dex_retargeting | OpenXR 손 관절 → 로봇 손가락 관절 각도 (Bench2Dex, Manus용 튜닝) |
| `manus_quest_isaaclab_device.py` | Isaac Lab | `ManusDevice`: 글러브 손가락 + 손목(컨트롤러 / 손 추적 / 고정) → 손 추적과 같은 형식. START 때 회전 보정·recenter·리타게터 기준점 설정 |
| `dexverse_integration.py` | DexVerse | `--teleop_device manus` 연결: Bench2Dex 손가락 리타게터, `add_manus_device(_from_env)`, `VR_TELEOP_*` 환경변수 |

**넣을 것**: 텔레옵 루프 안에서 import되어 쓰이는 코드. **넣지 말 것**: 사람이 실행하는 프로그램(`tools/`), 기기 하나만의 처리(`devices/`).

**START 흐름** (`ManusDevice`): ① 회전 캘리브레이션이 없거나 `calibrate_rotation`이면 손을 로봇 손 방향으로 보고 계산
② XR 기준점을 옮겨 내 손목을 로봇 손목에 겹침 ③ 몇 프레임 뒤 START 콜백(리타게터가 손목 기준점을 잡고 녹화 시작).
로봇 손목 pose는 리타게터의 `robot_wrist_pose_w` 속성으로 받는다(DexVerse 스크립트가 매 스텝 넣어 줌).
