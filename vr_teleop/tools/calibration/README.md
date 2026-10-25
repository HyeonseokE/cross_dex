# tools/calibration/ — 글러브-컨트롤러 마운트 캘리브레이션

손등 마운트에 단 Quest 컨트롤러 pose를 손목 pose로 바꾸는 두 고정 변환을 구한다. 마운트를 새로 끼우거나 크게 틀어졌을 때만
다시 한다(한 번 구한 값을 여러 사람이 써도 되고, 움직임이 이상하면 다시 맞춘다).

| 파일 | 종류 | 내용 |
|---|---|---|
| `pivot_solver.py` | 계산 (numpy) | 피벗 오프셋: 손목을 고정하고 돌린 컨트롤러 pose들로 `t_i + R_i r = p`를 최소제곱 → 컨트롤러 기준 손목 회전 중심 `r` |
| `mount_calibration.py` | 계산·파일 (numpy) | 회전 오프셋 `R_ctrlᵀ R_hand`, 캘리브레이션 JSON(`MountCalibrationFile`: 좌·우 항목, 저장할 때 이전 파일을 `history/`로 백업). 텔레옵이 이 파일을 읽는다 |
| `calibrate_pivot.py` | VR 앱 `[Isaac Sim]` | 피벗 기록 + 계산 + 저장 |
| `calibrate_rotation.py` | VR 앱 `[Isaac Sim]`, 패스스루 | 회전 오프셋 측정 + 저장 |

## 순서 (오른손 예, DexVerse 기준 실행)

```bash
cd /workspace/DexVerse && source /workspace/dexverse_env.sh          # VR_TELEOP_CALIB 설정됨
# 1) 피벗: 손목을 책상 모서리에 고정, B/Y로 기록 시작 → 20~30초 꺾고 비틀기 → A/X (기록 종료·계산·저장)
scripts/teleop_tools/run_teleop.sh vr_teleop/vr_teleop/tools/calibration/calibrate_pivot.py --side right
# 2) 회전: AR(패스스루)로 접속, B/Y로 초록 손 표시 → 실제 손을 겹치고 노란 점을 초록 큰 점에 → A/X (0.5초 정지, 저장 후 종료)
scripts/teleop_tools/run_teleop.sh vr_teleop/vr_teleop/tools/calibration/calibrate_rotation.py --side right
```

판정: 피벗 RMS 5 mm 이하·가장 좁은 회전 범위 30° 이상, 회전은 저장 로그의 손목-목표 거리가 몇 cm 이내·캡처 중 손 움직임(`hand moved up to`)이 몇 도 이내.

## JSON 형식

```json
{"description": "...",
 "right": {"offset_ctrl_m": [x, y, z], "rms_mm": 24.3, "rotation_span_deg": [...], "pivot_date": "...", "pivot_source": "history/pivot_right_....npz",
           "mount_rot_ctrl_to_wrist": [[...], [...], [...]], "mount_rot_date": "...", "mount_rot_method": "calibrate_rotation.py (...)"},
 "left": {...}}
```
손목 위치 = `t_ctrl + R_ctrl · offset_ctrl_m`, 손목 좌표(OpenXR 규약) = `R_ctrl · mount_rot_ctrl_to_wrist`.

## 알아 둘 것

- 피벗만으로는 회전을 알 수 없다(손목을 고정한 조건에 회전 차이가 드러나지 않음). 그래서 회전은 별도로 잰다.
- 회전 앱은 JSON에 이미 있는 회전값을 쓰지 않는다(이전 결과 위에 쌓이지 않게). 저장하면 바로 종료되므로, 결과는 텔레옵에서 확인한다.
- 컨트롤러와 손 추적이 동시에 오지 않는 경로라, 손 추적으로 회전을 자동 측정하는 방식은 쓸 수 없다.
- 피벗을 다시 재면(마운트를 다시 끼웠다면) 회전도 다시 잰다.
