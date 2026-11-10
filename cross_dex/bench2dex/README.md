# cross_dex.bench2dex — Bench2Dex에 Manus 장갑 + Quest 컨트롤러 텔레옵 연결 (2026-11-11)

DexVerse 수집에 쓰는 장비·캘리브레이션·손가락 리타게팅(`cross_dex.teleop`)을 그대로 Bench2Dex(`main.py --teleop --collect`)에
연결한다. Bench2Dex fork(`HyeonseokE/bench2dex_retarget`, 브랜치 `teleop/manus-quest`)에는 `main.py`의 `--teleop-device manus-quest`·`--robot-key`
분기만 추가했고, 나머지(팔 IK·collector·homing·HDF5)는 Bench2Dex 코드 그대로다.

```
Manus 브리지(TCP 47000) ─► 25 keypoints + 손목 쿼터니언 ─► GloveWristFrame
Quest 컨트롤러 (OpenXR, XrCfg anchor 적용된 sim 월드) ─► HandMountedController(캘리브레이션 JSON) ─► 손목 (pos, R)
glove_to_openxr_joints ─► OpenXR 26관절 ─► HandSample          adapters/manus_quest_hand_source.py
   ├ 손가락: Bench2DexHand (cross_dex hand_cfgs, Manus 튜닝)       adapters/manus_quest_teleop.py (XrTeleopController 상속)
   └ 손목: XrIsaacLabBridge — START/homing 시점 anchor 기준 상대 매핑 → ArmIKController (Bench2Dex 그대로)
```

## DexVerse와 다른 점 (적용 환경 차이)

| 항목 | DexVerse | Bench2Dex |
|---|---|---|
| 로봇 | 떠 있는 손 (손목 6D 관절) | 팔 + 손 (`--robot-key`, 기본 UR5+Wuji). 손목은 팔 IK 목표 |
| 손목 매핑 | 절대: START 때 XR anchor를 옮겨 사람 손목 = 로봇 손목 (recenter) | 상대: anchor 시점의 (사람 손목, 로봇 손바닥) 기준 변위·회전 (`--xr-pos-scale`). START/STOP/RESET마다 재anchor |
| 컨트롤러→손 회전 캘리브레이션 | START 때 로봇 손 자세로 자동 계산 가능 | 저장된 캘리브레이션 JSON만 사용 (없으면 경고). 캘리브레이션 앱은 DexVerse 쪽에서 먼저 |
| 손가락 설정 | `cross_dex/teleop/retarget` | 같은 파일 (fork의 `teleop/hand_cfgs`는 쓰지 않음). Inspire는 `rh56dfx.yml` right 섹션이 DexVerse URDF 기준이라 **Bench2Dex UR5+RH56DFX는 별도 확인 필요** |
| 에피소드 조작 | A/X 버튼·패널 Play, 자동 저장 | 헤드셋 START(홈 복귀 후 녹화)·STOP(홈 복귀 후 저장)·RESET(폐기). `--xr-auto-stop-success-steps N` |
| 저장 형식 | `dexverse_trajectory` pkl → LeRobot | Bench2Dex HDF5(origin, 상태만) → `replay.py --enable-rgb`로 카메라·촉각 렌더링 |
| 카메라 | 녹화 중 렌더링 | XR 세션 안에서는 collector 카메라 금지(첫 프레임 멈춤) → replay 단계에서 |
| 시드·장면 | episode_seed (에피소드 번호 고정) | Bench2Dex `--seed`(수집 기본 10000)·`--enable-generalization` |

## bs1 실행

```bash
# 터미널 1: CloudXR 런타임 (DexVerse와 공용, 호스트당 1개)
bash /workspace/DexVerse/scripts/teleop_tools/start_cloudxr_runtime.sh
# 터미널 2
source /workspace/bench2dex_env.sh        # DexVerse env 재사용 (Isaac Sim 5.1·Isaac Lab 2.3.2·Flask·h5py 포함), B2D_ROOT=/workspace/bench2dex
/workspace/cross_dex/cross_dex/bench2dex/adapters/b2d_session.sh teleop --task scenes/06_fruit_bowl_loading.yaml          # 조작만
/workspace/cross_dex/cross_dex/bench2dex/adapters/b2d_session.sh record --task scenes/06_fruit_bowl_loading.yaml \
    --robot-key multi_ur5_shadow_hand_with_flange                                                                        # 녹화 (--collect)
```
헤드셋: 클라이언트 페이지 → Connect → Enter VR, Video encoding **H265**. 로그 `/workspace/logs/b2d_teleop/<날짜>/`.
에셋: `/workspace/assets/bench2dex/{Robots_p,Objects,Background,scenes,textures,URDF-Zoo}` (`/workspace/dex2bench_dataset` 링크).

## 검증 상태

- 오프라인(헤드셋·Isaac 없음, `adapters/test_manus_quest_offline.py`, 2026-11-11): 가짜 브리지 + 가짜 컨트롤러 → HandSample(손목 = 컨트롤러 + 캘리브레이션 오프셋), Wuji 20관절·Shadow 24관절 리타게팅 OK.
- Isaac 기동(헤드셋 없음)·실제 헤드셋 조작·HDF5 녹화·replay 재현: 미확인 (다음 단계).
