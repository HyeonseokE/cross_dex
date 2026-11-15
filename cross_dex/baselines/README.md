# cross_dex/baselines — 비교군(베이스라인) 래퍼

다른 논문 방법을 DexVerse·DexJoCo-X 실험에 붙이는 **우리 코드**. 저자 코드는 고치지 않고 `codes/dexverse/<베이스라인>`에 그대로 둔다
(원본 `main`, 작업 브랜치 `dexverse`). 저자 저장소에는 push할 수 없으므로 래퍼 패키지를 여기(push 가능한 cross_dex)에 둔다 (2026-11-15 이동).

| 폴더 | 패키지 | 방법 | 저자 코드 (그대로 사용) |
|---|---|---|---|
| `xlvla/dexverse_xlvla/` | `dexverse_xlvla` | XL-VLA 공유 잠재 행동 공간: 손 모델(`hands.py`, DexJoCo-X `hands_dexjoco.py`), 손목 표현(`wrist.py`), 잠재 공간 학습·평가, 데이터 변환(`make_dataset.py`), 평가(`run_eval.py`, `eval_mt.sh`), pi0.5(`pi05/`) | `codes/dexverse/DexLatent` (`HandLatent/`), `codes/dexverse/openpi` |
| `lad/dexverse_lad/` | `dexverse_lad` | LAD(Latent Action Diffusion) 대조 학습 잠재 공간: 쌍 데이터(`build_pairs*.py`), 학습(`train_cam.py`), pi0.5용 손가락 공간(`pi05_space.py`, `make_pi05_dataset.py`) | `codes/dexverse/robot_clip` (`robot_clip/`) |

## 실행 경로 (바뀌지 않음)

옛 위치에는 상대 링크가 있다. 기존 명령·PYTHONPATH·서버 경로가 그대로 동작한다.

```
codes/dexverse/DexLatent/dexverse_xlvla  ->  ../../cross_dex/cross_dex/baselines/xlvla/dexverse_xlvla
codes/dexverse/robot_clip/dexverse_lad   ->  ../../cross_dex/cross_dex/baselines/lad/dexverse_lad
```

- XL-VLA: `cd /workspace/codes/dexverse/DexLatent && python -m dexverse_xlvla.<모듈>` (cwd의 `HandLatent`를 함께 import).
- LAD: `cd /workspace/codes/dexverse/robot_clip && python -m dexverse_lad.<모듈>`; pi0.5 변환은 `PYTHONPATH=<DexLatent>:<robot_clip>`.
- 패키지 이름을 `cross_dex.baselines.*`로 바꾸지 않은 이유: 저자 패키지(`HandLatent`, `robot_clip`)와 같은 폴더에서 import되는 구조이고,
  DexVerse·DexJoCo-X 손 정의를 한 패키지에서 같이 쓰기 때문.

## 규칙

- 공통 코드(DexVerse·LeRobot·평가)는 여기서 고치지 않는다. 연결은 래퍼(상속·교체)로 한다 (hscho_share `CLAUDE.md` "베이스라인·비교군 코드 구현").
- 논문·저자 코드와 다르게 한 점은 실험 bundle README에 적는다.
- `xlvla/dexverse_xlvla/dexjoco_urdf/`(DexJoCo-X MuJoCo 모델에서 만든 URDF)는 라이선스 확인 전이라 git에 넣지 않았다 (서버로는 sync됨).
