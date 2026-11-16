# cross_dex/ours — 우리 방법·변형

다른 논문의 재현(`../baselines/`)과 섞지 않는다 (2026-11-16 사용자 지시: "원본 모듈이랑 우리가 구현한 것이랑 구분").
`baselines/`에는 원 논문 재현과 DexVerse·DexJoCo-X 적용만 두고, 우리 손실·구조는 여기서 재현 코드를 상속하거나 훅으로 붙인다.
재현 코드의 기본 동작은 바뀌지 않는다.

| 모듈 | 무엇 | 바탕 (재현 코드, 그대로 호출) |
|---|---|---|
| `xlvla_local.py` | XL-VLA + 국소 일관성 손실 (미세 차이 보존) | `dexverse_xlvla.train_latent` (트레이너 상속, 옵티마이저 래퍼) |
| `lad_cover.py` | LAD + 관절 한계 정규화 + XL-VLA 무작위 자세 손실 (커버리지) | `dexverse_lad.train_cam` (`Hooks`) |

우리 청크 잠재 공간(`../latent/chunk_latent/`)도 우리 방법이다 (처음부터 따로 있었음).

이름 규칙: 표·HF·실험 기록에서 재현은 원 논문 이름 그대로 ("XL-VLA", "LAD"), 우리 것은 "ours: ..."로 쓴다.
실험은 `experiments/<server>/RSS2027/ablation/` (재현은 `baseline-reproduce/`).
