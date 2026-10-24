# devices/metaquest/ — Meta Quest 3

| 파일 | 내용 |
|---|---|
| `quest_openxr_isaaclab.py` `[Isaac Sim only]` | Isaac Sim OpenXR(CloudXR 6 + IsaacTeleop 웹 클라이언트) 경유 입력. `QuestXR`: 컨트롤러 pose·버튼, 머리 pose, 손 추적 관절, XR 기준점 이동, 헤드셋 메뉴 콜백. `TrackedPose`: 접속 전 자리표시 값과 실제 추적 구분. `request_ar_session`: XR 세션 시작 |

- 좌표는 시뮬레이션 월드 기준(XR 기준점 적용). 컨트롤러 pose 행: `[x, y, z, qw, qx, qy, qz]`, 버튼: `BUTTONS`
  (`trigger`, `squeeze`, `primary`=A/X, `secondary`=B/Y).
- 이 경로에서는 헤드셋이 접속할 때의 입력 모드에 따라 **컨트롤러나 손 추적 중 하나만** 온다.
- 다른 XR 런타임·시뮬레이터에서는 같은 메서드를 가진 클래스를 새로 만들면 나머지 코드는 그대로 쓴다.
- 손등 마운트(컨트롤러 → 손목)는 기기 하나의 일이 아니라서 `teleop/hand_mounted_controller.py`와 `tools/calibration/`에 있다.
