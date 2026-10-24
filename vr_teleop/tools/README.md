# tools/ — 사람이 실행하는 도구

| 파일 | 내용 |
|---|---|
| `run_manus_bridge.sh` | Manus 브리지 실행 (`--fake`: 합성 손) |
| `dexverse_session.sh` `[DexVerse]` | 텔레옵 / 녹화를 표준 설정으로 실행. 브리지가 없으면 띄우고, 로그는 `/workspace/logs/teleop/YYYYMMDD/` |
| `check_fingers_headless.py` `[DexVerse]` | VR 없이 손가락 리타게팅 점검 (손목 고정, 영상·관절 CSV 저장) |
| `calibration/` | 글러브-컨트롤러 마운트 캘리브레이션: 계산·파일 형식·VR 앱 |

**넣을 것**: 직접 실행하는 프로그램과, 여러 기기에 걸친 캘리브레이션. **넣지 말 것**: 텔레옵 루프 코드(`teleop/`).
