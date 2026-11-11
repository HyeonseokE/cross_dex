# chunk_paired — 손 간 짝 청크 데이터셋 (Feix 파지 분류 전체)

사람 손 동작과 5지 로봇 파지를 같은 표현으로 바꾸고, Feix 파지 종류 라벨을 붙인 뒤, DexVerse 텔레옵 변환기로 각 5지손에 옮겨
정책 주기(60 Hz)·H = 16 청크로 자른다. 손 사이 짝 = 같은 원본 동작의 같은 시간 구간.
설계·단계·검증 기준: 실험 묶음 `experiments/a4/RSS2027/ablation/dexverse-chunk-latent/DATASET_PIPELINE.md`, Feix 대응표 `FEIX_MAPPING.md`.

실행: `PYTHONPATH=/workspace/codes/cross_dex <env python> -m cross_dex.dataset.chunk_paired.<모듈>` (공통 환경에 설치하지 않고 경로로만 쓴다).

| 모듈 | 환경 | 역할 |
|---|---|---|
| `extract_dexcanvas` | lerobot (pyarrow) | DexCanvas parquet → shard별 npz (관절점 21개, MANO 손목 회전, 라벨) |
| `retarget_dexcanvas` | main (Isaac Lab) | 텔레옵 변환기(`SimpleRelativeRetargeter`) 그대로 → 손별 손가락 행동·URDF 관절값·오차 |
| `check_ranges` | lerobot | 리타게팅 관절 범위 vs 실제 텔레옵 데모 범위 |
| `transition_chunks` | dexlatent (`dexverse_xlvla`) | 손마다 같은 의미 자세(편 손·엄지 핀치 4종) 사이 전이 청크, 전이 시간 = 사람 동작 시간 분포 |
| `motion_timing` | lerobot | 사람 손 오므림·펴기 시간 분포 (합성 전이 시간의 기준) |
| `viewer_server` + `viewer.html` | lerobot | 같은 에피소드의 손별 리타게팅 결과 웹 뷰어 (로봇 메시 + 사람 관절점) |
| `dexonomy_viewer` + `.html` | openpi (mujoco) | Dexonomy(Shadow) 파지 웹 뷰어, pre-grasp → grasp → squeeze 보간 재생 |

데이터 위치(서버 로컬): `/workspace/datasets/dexverse/{human_motion/dexcanvas*, grasp_taxonomy/{dexonomy,hograspnet}, chunk_latent}`.
