# cross_dex — DexVerse·Bench2Dex 공용 코드

cross-dex 프로젝트에서 두 벤치마크(DexVerse, Bench2Dex)가 함께 쓰는 **우리 코드**를 한 곳에 둔다.
벤치마크 fork(`codes/dexverse/DexVerse` dev, `codes/bench2dex_*`)에는 원본 + 최소 패치만 남기고,
장비·리타게팅·연결부·파이프라인은 여기서 `pip install -e`로 가져다 쓴다.

```
cross_dex/                      (import root)
  teleop/                       VR 텔레옵: Manus 장갑 + Quest 3 컨트롤러·헤드셋, 캘리브레이션, 손가락 리타게팅
                                — 시뮬레이터 무관 (README.md에 사용 순서)
    common/  devices/{manus,metaquest}/  retarget/  hand_mounted_controller.py  tools/{calibration,hand_viewer}
  dexverse/
    adapters/                   DexVerse 연결부: Isaac Lab 장치, record_demos 통합, 성공 패널, 세션 스크립트
  bench2dex/
    adapters/                   Bench2Dex 연결부 (예정: Manus+컨트롤러 → XrHandSource 호환 입력, 세션 스크립트)
```

예정: `retarget/`(SPIDER cross-embodiment, 현재 `bench2dex_retarget/retarget`), `pipeline/`(pkl→LeRobot·HF·노션,
현재 `_setup/bs1_pipeline`), `baselines/`(LeRobot DP·DexLatent 래퍼, 현재 `DexVerse/baselines`).

## 설치

```bash
pip install --no-deps -e /workspace/cross_dex        # 시뮬레이터 환경에 (예: /workspace/envs/dexverse/main)
```
필요한 외부 것은 환경별: numpy·scipy(기본), dex_retargeting+pinocchio(손가락 리타게팅), Isaac Sim/Isaac Lab(장치·캘리브레이션),
dexverse 패키지(DexVerse adapters), Manus Linux SDK(`$MANUS_SDK_PY`), Bench2Dex URDF-Zoo(`$URDF_ZOO_DIR`).

## 벤치마크별 실행

- **DexVerse**: `cross_dex/dexverse/adapters/dexverse_session.sh record --task <Task> --robot_type <rt> --num_demos 50`
  (DexVerse 레포의 `scripts/teleop_tools/dexverse_session.sh`는 이 스크립트로 넘기는 래퍼).
  `record_demos.py --teleop_device manus`가 `cross_dex.dexverse.adapters.dexverse_integration`을 부른다.
- **Bench2Dex**: 준비 중 (`bench2dex/adapters/`).

## 규칙

- 벤치마크 원본 코드는 고치지 않고 adapters에서 감싼다. 벤치마크 fork에 남는 패치는 "cross_dex를 부르는 한 줄"로 최소화.
- 손 리타게팅 설정(`teleop/retarget/{angle_cfgs,bench2dex/hand_cfgs}`)은 여기 한 벌만 둔다 (Bench2Dex fork의 `teleop/hand_cfgs`는 쓰지 않는다).
- 이력: `teleop/`는 DexVerse `vr_teleop/`(dev, 2026-11-10까지 53커밋)를 `git subtree split`으로 옮긴 것.
