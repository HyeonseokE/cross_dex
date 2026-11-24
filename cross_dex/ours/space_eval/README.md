# 공유 행동 공간 평가 키트

손 여러 개가 함께 쓰는 잠재 행동 공간(공유 공간)을 아티팩트의 "전체 지표" 표와 같은 방법·같은 데이터로 수치화한다.
어댑터 파일 하나만 쓰면 어떤 공간이든 표에 있는 공간들과 나란히 비교할 수 있다.

- 결과 표 (아티팩트): https://claude.ai/artifact/FgKPCw8KNcJ3EX8BTm77U6
- 기준 문서: `experiments/a4/RSS2027/README.md`의 "공유 공간 정렬의 요구 조건", "손 간 쌍 뭉침 평가"
- 2026-11-19 작성

## 1. 빠른 시작

컨테이너(`/workspace` 구조) 안에서 실행한다.

```bash
# 1) 평가 데이터 받기 (HF private, 한 번만)
huggingface-cli download Dexverse-CSI/shared-action-spaces --include "space_eval_data/*" --local-dir /workspace/.tmp/hf_sas
export SPACE_EVAL_DATA=/workspace/.tmp/hf_sas/space_eval_data      # amp7에는 이미 /workspace/datasets/dexverse/lad/eval_space

# 2) 내 공간 어댑터 작성 (아래 3장, 예: adapter_xlvla.py)

# 3) 전체 지표 계산
cd /workspace/codes/cross_dex/cross_dex/ours/space_eval
ADAPTER_PY=<내 모델을 import할 수 있는 python> bash run_all.sh my_adapter.py <체크포인트> <이름> dv_right
```

마지막에 내 공간 한 줄과 기준 공간들(`reference_table.json`)을 같은 열로 출력한다. 결과 파일은 `$WORK`(기본 `/workspace/.tmp/space_eval_kit/<이름>`)에 남는다.
DexJoCo-X는 마지막 인자를 `dj_right`로 바꾼다.

## 2. 평가 데이터

HF `Dexverse-CSI/shared-action-spaces` (private) `space_eval_data/<세트>/`. 손 관절각은 rad, 손목 제외, 열 순서는 `inputs.npz`의 `names[손]`.

| 세트 | 내용 | 쓰는 지표 |
|---|---|---|
| `dv_right` | DexVerse 오른손 6손(shadow, allegro, inspire, leap, sharpa, wuji) 태스크 데모 프레임 `teleop_<손>`, 무작위·핀치 자세(`inputs_extra.npz`) | 같은 손 복원, 교차 방향, 해상도, 연속성 |
| `dv_right_demopair` | 원천 손마다 데모 프레임 400개(손끝 5 mm 이상 간격)를 나머지 손으로 DexPilot 리타게팅한 2,400 튜플 `pair_<손>`, 잔차 `pair_res_mm_<손>`, 원천 `pair_src` | 쌍 뭉침, 정밀도, 교차 재구성, 쌍 자기 |
| `dv_right_bundle` | 원천 손마다 기준 자세 80개 + 1~5 mm 이웃 4개 = 묶음 5개씩, 이웃은 기준 자세 해를 시작점으로 리타게팅. `pair_bundle` = 묶음 번호 | 손 간 해상도 |
| `dj_right*` | DexJoCo-X 오른손 7손, 위와 같은 구성 (데모 1,262 튜플, 묶음 235개) | 위와 같음 |

- 리타게팅: DexPilot 비용 (huber 0.03, project 0.03, eta1 1e-4, eta2 3e-2), 300 it, 2 starts. 만든 코드 `data_gen/` (`make_demo_pairs.py`, `make_bundle_pairs.py`).
- 기하 오라클 (DexPilot 거리로 짝 찾기, 참고 상한): DexVerse R@1 56.0 %, 중앙 순위 9.0, 손 간 해상도 64.9 % / DexJoCo-X 41.6 %, 10.9, 49.3 % (`data_gen/pair_res_demo.py`, `pair_res_b.py`).
- 세트는 다시 만들지 않는다. 모든 공간을 같은 프레임으로 잰다.
- `dv_right`의 쌍(`pair_*`)은 제3의 사람 데모(LAD 검증 쌍) 기반이라 외부 공개 금지. 저장소를 public으로 바꾸지 않는다.

## 3. 어댑터

```python
def load(ckpt, hands, joint_names, side, bench):   # hands: 손 이름, joint_names[손]: 열 순서, side "right", bench "dexverse"|"dexjoco"
    return model
model.encode(hand, q)  # q: torch.float32 [N, 관절 수] rad -> z [N, d]
model.decode(hand, z)  # z [N, d] -> q [N, 관절 수] rad (같은 열 순서)
```

- 정규화·관절 순서 변환은 어댑터 안에서 한다. 예: `adapter_xlvla.py` (XL-VLA 체크포인트, 표의 XL-VLA 계열과 같은 값).
- 손마다 z 차원이 같아야 한다 (손 간 교차 디코딩과 검색에 그대로 쓴다).
- 거리 기반 검색은 L2이므로 z 크기도 의미가 있다. 코사인 공간이면 어댑터에서 정규화한 z를 돌려준다.

## 4. 지표 ("전체 지표" 표 열)

손끝 오차는 손 FK(`dexverse_xlvla.hands`)로 구한 표준 손끝 위치 차이(mm). 값은 손 평균(검색 지표는 순서 있는 손 쌍 평균).

| 묶음 | 열 | 정의 | 세트 | 좋은 쪽 |
|---|---|---|---|---|
| 손 간 뭉침 | R@1 % | 손 A 쌍의 z가 손 B의 모든 쌍 z 중 짝을 가장 가깝게(L2) 찾는 비율 | demopair | 높게 |
| | 중앙 순위 | 짝의 순위 중앙값 (1 = 최고) | demopair | 낮게 |
| | 정밀도 R@1 % | 리타게팅 잔차 < 5 mm인 쌍만의 R@1 | demopair | 높게 |
| | 손 간 해상도 % | 같은 묶음 5개(1~5 mm 차이) 안에서 짝을 맞히는 비율 (우연 20 %) | bundle | 높게 |
| 손 간 복원 | 교차 재구성 mm | 손 A의 z를 손 B로 디코딩한 것 vs 리타게팅 짝 | demopair | 낮게 |
| | 데모·핀치·무작위 교차 ° | 손 A 엄지-손가락 손끝 선과 손 B로 옮긴 선의 방향 차이 | dv_right | 낮게 |
| 같은 손 복원·커버리지 | 데모 자기 mm | 데모 프레임 인코딩→같은 손 디코딩 | dv_right | 낮게 |
| | 무작위 자기 mm | 관절 한계 안 균일 자세 (커버리지) | dv_right | 낮게 |
| | 쌍 자기 mm | 평가 쌍 자세(데모 + 리타게팅된 자세)의 같은 손 복원 | demopair | 낮게 |
| 해상도·매끄러움 | 해상도 % / mm | 데모 q와 q+d(관절 RMS 1°)의 출력 변화가 d를 재현하는지: 상대 오차 / 손끝 오차 | dv_right | 낮게 |
| | 연속성 mm | z에 잡음(z 표준편차의 5 %)을 더했을 때 출력 변화. 해상도와 함께 본다 | dv_right | 참고 |

## 5. 파일

| 파일 | 하는 일 |
|---|---|
| `run_all.sh` | 아래 4단계를 차례로 실행 |
| `run_space.py` | 어댑터로 디코딩(`decoded_<이름>.npz`)과 z 저장 |
| `../../baselines/lad/dexverse_lad/eval_space.py` | `metrics`: 복원·교차·해상도·연속성 계산 → `space_<이름>.json` (dexlatent 환경) |
| `pair_metrics.py` | 쌍 뭉침·정밀도·손 간 해상도 (numpy만) |
| `full_row.py` | 표 한 줄 조립 + 기준 공간과 함께 출력 |
| `reference_table.json` | 아티팩트 표의 기준 공간 값 (2026-11-19) |
| `data_gen/` | 평가 세트 생성·오라클 코드 (참고, 다시 돌릴 필요 없음) |

환경: 디코딩은 내 모델의 환경(`ADAPTER_PY`), `metrics`는 dexlatent 환경 + 손 URDF(`DEXVERSE_ROBOT_AGENTS`, 기본 `/workspace/assets/dexverse/robot_agents`).
