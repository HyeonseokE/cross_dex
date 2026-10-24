# vr_teleop — Manus 글러브 + Meta Quest 3 VR 텔레옵

손가락은 **Manus Metagloves Pro**, 손목 6D pose는 **손등 마운트에 단 Quest 3 컨트롤러**, 화면은 **Quest 3 VR**로
시뮬레이션 속 로봇 손을 조종하고 데모를 녹화하기 위한 패키지. 기기 처리, 캘리브레이션, 리타게팅을
시뮬레이터와 분리해 두어 **이 폴더를 다른 시뮬레이션 프로젝트로 옮겨 쓸 수 있게** 했다.
현재 연결된 시뮬레이터는 Isaac Lab / DexVerse.

## 데이터 흐름

```
[Manus 글러브] ─동글─ devices/manus/bridge (SDK 서버, TCP 47000)
                              │ 25관절 + 손목 쿼터니언
                              ▼
                     devices/manus/glove_skeleton.py ── 손가락 모양 (글러브 손목 좌표계)
[Quest 3 컨트롤러] ─ devices/metaquest/quest_openxr_isaaclab.py ── 컨트롤러 6D pose
                              │
                     teleop/hand_mounted_controller.py ── 손목 6D pose
                              │   (pivot offset + mount rotation: tools/calibration 결과 JSON)
                              ▼
                     OpenXR 손 관절 26개 (common/openxr_hand.py 규약; 헤드셋 손 추적과 같은 형식)
                              │
                     teleop/retarget ── 로봇 손가락 관절 각도 (Bench2Dex, Manus용 튜닝)
                     teleop/manus_quest_isaaclab_device.py ── Isaac Lab 장치 (START 정렬, recenter)
                     teleop/dexverse_integration.py ── DexVerse --teleop_device manus
```

## 폴더 구성과 규칙

| 폴더 | 역할 | 들어가는 것 | 들어가면 안 되는 것 |
|---|---|---|---|
| `common/` | 모든 기기가 공유하는 규약 | OpenXR 손 관절 이름·손목 좌표 규약·손 템플릿·뼈 연결, 회전 유틸. numpy/scipy만 | 특정 기기·시뮬레이터 코드 |
| `devices/<기기>/` | **그 기기 하나만으로** 하는 일 | 기기 SDK/드라이버 연결, 기기 원시 데이터 → 표준 형식 변환 | 다른 기기와 엮인 처리(마운트, 캘리브레이션), 로봇·시뮬레이터 로직 |
| `devices/manus/` | Manus 글러브 | SDK 브리지(서버·클라이언트), 글러브 스켈레톤 → OpenXR 관절 | 손목 위치(글러브는 위치 추적 없음) |
| `devices/metaquest/` | Quest 3 헤드셋·컨트롤러 | XR 런타임에서 컨트롤러·머리·손 추적 pose, 버튼, XR 기준점 | 마운트 오프셋, 글러브 데이터 |
| `teleop/` | 기기를 조합해 로봇을 움직이는 실행 중 코드 | 손에 단 컨트롤러 → 손목, 손가락 리타게팅, 시뮬레이터 장치, 프레임워크 연결 | 사람이 직접 실행하는 프로그램 |
| `tools/` | 사람이 실행하는 도구 | 브리지 실행, 점검, 캘리브레이션 앱과 그 계산·파일 형식 | 텔레옵 루프 코드 |

파일 머리말의 `[Isaac Sim only]` / `[Isaac Lab only]` / `[DexVerse only]` 표시가 없는 모듈은 **시뮬레이터 무관**
(numpy/scipy, 리타게팅은 dex_retargeting)이다.

## 다른 시뮬레이터로 옮길 때

| 그대로 쓰는 것 | 새로 쓰는 것 |
|---|---|
| `common/`, `devices/manus/` 전체, `teleop/hand_mounted_controller.py`, `teleop/retarget/`, `tools/calibration/` 계산·파일 형식(`pivot_solver.py`, `mount_calibration.py`) | ① VR 입력: `devices/metaquest/quest_openxr_isaaclab.py`와 같은 메서드(`poll/controller/head/move_anchor/add_callback`)를 가진 클래스 ② 시뮬레이터 장치: `teleop/manus_quest_isaaclab_device.py` 같은 연결부 ③ 캘리브레이션 앱의 화면 부분 |

캘리브레이션 JSON은 시뮬레이터와 무관하게 그대로 쓴다.

## 설치 (1회)

```bash
pip install --no-deps -e /workspace/DexVerse/vr_teleop      # 시뮬레이터 환경(예: /workspace/envs/dexverse/main)에
```
필요한 외부 것: Manus Linux SDK(`$MANUS_SDK_PY`, 기본 `/workspace/manus_sdk/Python`), Bench2Dex URDF-Zoo(`$URDF_ZOO_DIR`),
CloudXR 런타임(DexVerse `scripts/teleop_tools/start_cloudxr_runtime.sh`), Quest와 서버가 같은 Tailscale.

## 사용 순서 (DexVerse)

```bash
source /workspace/dexverse_env.sh            # VR_TELEOP_CALIB 등 기본값 포함
cd /workspace/DexVerse
# 1) 터미널 1: CloudXR 런타임 (Quest 접속용)
CXR_ENDPOINT_IP=<서버 Tailscale IP> scripts/teleop_tools/start_cloudxr_runtime.sh
# 2) 캘리브레이션 (마운트를 새로 끼웠을 때만): tools/calibration/README.md
scripts/teleop_tools/run_teleop.sh vr_teleop/vr_teleop/tools/calibration/calibrate_pivot.py --side right
scripts/teleop_tools/run_teleop.sh vr_teleop/vr_teleop/tools/calibration/calibrate_rotation.py --side right
# 3) 텔레옵 / 녹화 (Manus 브리지는 없으면 자동 실행)
vr_teleop/vr_teleop/tools/dexverse_session.sh teleop --task Dexverse-GraspCup-v1
vr_teleop/vr_teleop/tools/dexverse_session.sh record --task Dexverse-GraspCup-v1 --num_demos 20
```
Quest: 컨트롤러 모드(손 추적 끄기) → `https://nvidia.github.io/IsaacTeleop/client` → 서버 IP, 포트 48322 → Connect.
캘리브레이션 앱은 AR(패스스루), 텔레옵은 VR로 들어간다. 프로그램을 다시 띄울 때마다 Disconnect → Connect.

**START**: 로봇은 헤드셋 메뉴 START 전에는 움직이지 않는다. 손을 로봇 손 근처·같은 방향(책상에서 약 20 cm 위)에 두고
START → XR 기준점을 옮겨 내 손을 로봇 손에 겹치고(recenter), 그때부터 변화량을 따라온다. 어긋나면 STOP → 맞추기 → START.
`record` 모드는 성공 조건이 유지되면 저장·리셋하고, 다음 데모도 START로 시작한다.

## 환경변수 (DexVerse 연결, `teleop/dexverse_integration.py`)

| 변수 | 값 | 기본 |
|---|---|---|
| `VR_TELEOP_WRIST` | `controller` / `handtracking` / `fixed` | `fixed` (`dexverse_session.sh`는 `controller`) |
| `VR_TELEOP_CALIB` | 캘리브레이션 JSON | 없음 (`dexverse_env.sh`에서 `.../calibration/current.json`) |
| `VR_TELEOP_FINGERS` | `bench2dex` / `dexverse` (Quest 손 추적용 DexVerse YAML) | `bench2dex` |
| `VR_TELEOP_AUTO_START` | `1`이면 손 데이터가 들어오자마자 시작 | `0` (헤드셋 START) |
| `VR_TELEOP_CALIBRATE_ROTATION` | `1`이면 START 때 로봇 손 방향 기준으로 회전을 계산·저장 (보조 방법) | `0` |

## 데이터 위치 (bs1)

| 무엇 | 위치 |
|---|---|
| 캘리브레이션 | `/workspace/datasets/dexverse/teleop/calibration/current.json → default/controller_wrist_offset.json`, 이전 값·피벗 원본 `default/history/` |
| 녹화 pickle | `/workspace/datasets/dexverse/teleop/trajectories/<카테고리>/<태스크>/*.pkl` (`DEXVERSE_LEROBOT_DIR`) |
| 로그 | `/workspace/logs/teleop/YYYYMMDD/` |
| Manus SDK / 글러브 설정 | `/workspace/manus_sdk`, `/workspace/manus_config` (`/root/.config/Manus` 링크) |

녹화 pickle에는 로봇 기준 데이터(action, 장면 상태)와 `provenance`(git commit, 실행 명령, `VR_TELEOP_*`·`DEXVERSE_*` 환경변수)가
저장된다. 레이아웃 복원은 `initial_state`·`task_state`로 하고, 그 환경을 만든 코드는 commit으로 되찾는다.

## 트러블슈팅

| 증상 | 확인 |
|---|---|
| 브리지 `gloves=[]` | 글러브 전원·절전, Windows MANUS Core에서 페어링 |
| CloudXR 로그 `no controller registered` | Quest가 손 추적 모드로 접속 → 컨트롤러 모드로 다시 Connect |
| 컨트롤러·손 추적이 동시에 안 옴 | 이 CloudXR/WebXR 경로의 한계 (둘 중 하나만 온다) |
| 빨간 점이 컨트롤러 방향을 봄 | 회전 캘리브레이션이 없음 → `calibrate_rotation.py` |
| 손목만 돌렸는데 로봇 손이 원을 그림 | 피벗 오프셋 부정확 → `calibrate_pivot.py` |
| 이전 캘리브레이션 앱 뒤 컨트롤러 데이터가 안 옴 | 이전 프로그램이 VR 세션을 잡고 있음 → 종료 후 Disconnect → Connect |
| 시뮬레이션 책상이 낮음 | START 때 손을 책상에서 약 20 cm 띄운다 |
