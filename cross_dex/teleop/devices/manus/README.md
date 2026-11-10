# devices/manus/ — Manus 글러브

| 파일 | 실행 위치 | 내용 |
|---|---|---|
| `bridge/manus_sdk_reader.py` | 동글이 꽂힌 곳 | Manus SDK(Integrated 모드)로 글러브 raw skeleton 25노드(WebXR 순서), 손목 노드 쿼터니언, ergonomics 각도를 읽음 |
| `bridge/manus_bridge_server.py` | 동글이 꽂힌 곳 | 위 데이터를 TCP(기본 127.0.0.1:47000)로 초당 최대 100회 송출. `--fake`는 글러브 없이 합성 손 |
| `bridge/manus_bridge_client.py` | 아무 프로세스 | 최신 글러브 데이터를 받는 클라이언트 (표준 라이브러리 + numpy). 직접 실행하면 받는 값을 출력 |
| `glove_skeleton.py` | 텔레옵 쪽 | 글러브 손목 좌표계(손목 노드 축을 손바닥 관절에 맞춰 OpenXR 규약으로 매핑), 손가락 모양을 원하는 손목 pose에 놓아 OpenXR 관절 26개로 변환 |

- 동글은 **한 프로그램만** 잡을 수 있다. SDK는 브리지 하나만 띄우고 나머지는 클라이언트로 받는다.
- 실행: `tools/run_manus_bridge.sh` (SDK 경로 `$MANUS_SDK_PY`). Linux에서는 Integrated 모드라 Core 없이 동작하지만
  페어링·펌웨어·캘리브레이션은 Windows MANUS Core에서 한다. 글러브 캘리브레이션은 `~/.config/Manus`에 저장된다.
- `bridge/manus_sdk_reader.py`, `manus_bridge_server.py`는 robo-viz/manus에서 가져왔고 손목 쿼터니언(`wq`) 송신을 추가했다.
- 글러브에는 위치 추적이 없으므로 손목 위치는 이 폴더에서 다루지 않는다.
