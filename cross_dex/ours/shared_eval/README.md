# cross_dex/ours/shared_eval — 평가 모델 공유 (GPU당 모델 1개, 여러 평가 세션이 묶어서 추론)

우리 평가 방식이다 (재현 코드 아님). 재현 코드(`baselines/xlvla/dexverse_xlvla/**`, `baselines/lad/**`)와 공통 코드
(DexVerse `baselines/lerobot_dp/eval_lerobot.py`, LeRobot, openpi)는 **고치지 않고 import·호출만** 한다.

```
batch_server.py      공통: 접속·세션 관리, 요청 모으기(gather)·묶음 추론, 세션별 시드 규칙(episode_seed), 시간 기록(jsonl)
dp/                  LeRobot DP (2026-11-16 baselines/lerobot_dp_shared에서 이동, 4c63185·0134843)
  shared_server.py     DP 모델 1개 (lerobot env) — 세션별 관측·행동 큐, 세션별 노이즈 생성기
  run_eval_dp.py       공통 평가기를 own|dedicated|shared 서버로 실행 + 시간 측정
  bench.sh measure_all.sh summarize.py check_equivalence.py
pi05/                pi0.5 (openpi JAX) — XL-VLA 잠재(pi05_latent)·관절(pi05_joint)·LAD(pi05_lad), 한 손·양손
  openpi_server.py     GPU당 1개 (openpi venv): 모델 보유, infer 요청을 모아 한 번의 batch sample_actions
  batched_policy.py    openpi Policy.infer와 같은 처리를 B개 행으로 (행마다 그 세션의 노이즈 키)
  session_server.py    세션마다 1개 (DexLatent venv): 재현 코드의 Pi05PolicyServer / Pi05BimanualPolicyServer 그대로 + 공유 서버 연결
  run_eval_shared.py   공통 평가기 실행 (main env), policy_server.py 대신 session_server.py를 띄움
  eval_shared.sh       관리자 (eval_mt.sh와 같은 칸·claim·결과 폴더, GPU당 서버 1개 + 세션 WORKERS개)
  bench.sh             같은 칸 N세션 측정 (dedicated = 옛 경로 / shared)
  check_batch.py       GPU 오프라인 확인 (B=1 동일성, batch vs 단독, B별 시간·메모리)
  summarize.py         처리량·지연·메모리 요약, --compare 행동 비교
tests/test_shared.py CPU 단위 시험 (가짜 정책)
```

## 무엇이 어디서 도나 (pi0.5)

| 프로세스 | 옛 경로 (`eval_mt.sh`) | 공유 경로 (`eval_shared.sh`) |
|---|---|---|
| Isaac (main env) | `run_eval.py` → `eval_lerobot.py`: 시뮬·렌더링·관측 | 같음 (`run_eval_shared.py` → 같은 `eval_lerobot.py`) |
| 세션 정책 (DexLatent venv) | `server.py` → `Pi05PolicyServer`: 상태 인코딩(손목 정규화 9 + VAE/LAD/관절 패딩), 청크 큐(replan 10), 디코딩(손목 IK, VAE/LAD 디코더) | `session_server.py` → **같은 클래스** (`_spawn`만 교체: 프로세스 대신 공유 서버에 연결) |
| openpi (openpi venv) | 칸마다 `serve.py` 1개 (~8.8 GB) | **GPU당 `openpi_server.py` 1개**: 입력 변환(이미지 224, 정규화, 상태 토큰 포함 토큰화)은 행마다 그대로, `sample_actions`만 batch |

잠재·관절·LAD 변환과 손목 처리는 전부 세션 쪽 재현 클래스가 하므로 바뀌지 않는다. 공유 서버는 손·태스크를 모른다
(관측 이미지·상태 벡터·프롬프트만 받음) → 한 GPU의 세션들은 **같은 체크포인트(같은 MODE·run)** 여야 한다.

## 묶음 추론·노이즈

- **모으기** (`batch_server.py`, DP와 공통): 요청 하나가 오면 이미 와 있는 것을 다 받고, `--gather_ms` 동안 또는 접속한 세션이 모두 기다릴 때까지 더 모은 뒤
  한 번에 추론. `GATHER_MS=1000`(기본) = lockstep(매 추론에 모든 활성 세션), `5` = greedy. `--max_batch`로 자른다.
  세션이 기다리는 동안 시뮬레이션은 멈춰 있다 (평가기가 동기식: 응답 전에는 `env.step` 안 함).
- **노이즈** (`batched_policy.py`): openpi `Policy.infer`는 호출마다 `self._rng, k = split(self._rng)`(시작 `key(0)`), 모델 안에서 `normal(k, (1,H,D))`.
  공유 서버는 세션마다 키를 따로 들고 같은 방식으로 split 하고, 행 i의 노이즈를 `normal(k_i, (1,H,D))`로 만들어 `noise=`로 넘긴다
  → 어떤 세션과 같이 묶이든 그 세션의 노이즈는 같다.
  - `NOISE=process`(기본): 세션 접속 때 `key(0)` → **옛 경로의 칸별 openpi 프로세스와 같은 노이즈 열** (같은 칸·같은 에피소드면 옛 결과와 직접 비교 가능).
  - `NOISE=episode`: 에피소드마다 `key(seed*100000 + 에피소드)` (`batch_server.episode_seed`, DP와 같은 규칙) → 칸 이력(앞 에피소드 길이)과 무관.
- DP 쪽은 그대로: 세션별 `torch.Generator`(cuda), 에피소드 시드 `seed*100000 + 에피소드`.

## 실행

```bash
# pi0.5, 서버 컨테이너에서 (eval_mt.sh와 같은 환경변수 + 아래)
R=<run> CK=<openpi step dir> LATENT=<latent.pt> WRIST_DIR=<dir> MODE=pi05_latent PI05_RUN=<run> PI05_REPO=<id> PI05_ADIM=41 \
  GPU=0 WORKERS=4 bash /workspace/codes/cross_dex/cross_dex/ours/shared_eval/pi05/eval_shared.sh
#   BIMANUAL=1 PI05_MAXTOK=360 LATENT_LEFT=...  / MODE=pi05_lad LAD_CKPT=...   (eval_mt.sh와 같음)
#   GATHER_MS(1000) MAX_BATCH(=WORKERS) NOISE(process|episode) PRECOMPILE(=MAX_BATCH) LOG_ACTIONS=1 STAGGER(90)
#   SHARED=0: 칸마다 재현 run_eval.py(옛 경로, 칸별 openpi) — 같은 관리자로 옛/새 비교
#   DRY_RUN=1: 서버 명령·칸별 server.json·평가 명령만 출력
# 측정: 같은 칸 N세션
PKL=<pickle> SERVER_JSON=<칸의 server.json> GPU=0 bash pi05/bench.sh <out> shared 4 0,1
python3 pi05/summarize.py <out>...;  python3 pi05/summarize.py --compare <옛 dedicated out> <shared out>...

# DP (이전과 같음, 경로만 dp/)
PKL=... CKPT=<pretrained_model> bash dp/bench.sh <out> shared 4 0,2
```

결과 폴더는 `eval_mt.sh`와 같다: `<R>/eval/final/<task>/<hand>/{server.json, eval.log, eval/{metrics.json, videos/}}`,
claim `.claim_<task>_<hand>`, `want`, `STOPPED`, `KEEP_CLAIMS`, 끝난 칸 건너뛰기. 추가: 칸별 `timing.jsonl`, `<R>/logs/shared_gpu<GPU>.{log,jsonl,pid}`, `eval_shared.log`.

## 옛 경로와 다른 점

- openpi 프로세스: 칸마다 1개 → GPU당 1개. 서버가 죽으면 관리자가 30초 안에 다시 띄우고, 그때 돌던 칸은 실패 → 재시도(옛 경로와 같은 2회 / STOP_ON_KILL).
- 모델 호출: `Policy.infer` (B=1, 노이즈를 jit 안에서 생성) → 같은 입력·출력 변환 + `noise=`를 준 batch `sample_actions`. 모델 구성은 `serve.py`와 같은 인자로 `create_trained_policy` (serve.py는 로드와 서빙이 한 함수라 import하지 않고 그 몇 줄을 같은 값으로 씀).
- 관리자를 다시 띄우면 서버 포트가 바뀌므로, 돌고 있던 공유 세션은 이어받지 않고 끝낸 뒤 그 칸을 다시 시작한다 (`SHARED=0`은 eval_mt.sh처럼 이어받음).
- 첫 요청 때 `PRECOMPILE`개 batch 크기를 미리 컴파일 (그 사이 모든 세션 대기, 세션 키는 안 씀).
- 세션 정책 쪽 `reset` 때 공유 서버에 `("reset", {"episode": i})`를 하나 더 보낸다 (process 모드에서는 무시).
- PyTorch openpi 체크포인트(`model.safetensors`)는 지원 안 함 (지금 pi0.5 체크포인트는 JAX).

## CPU에서 확인한 것 (2026-11-16, PC)

- `tests/test_shared.py` (`uv run --no-project --with "jax[cpu]" --with numpy python tests/test_shared.py`) 모두 통과:
  ① 공통 서버: 4세션 동시(lockstep, batch 4) == 각각 단독 ② 가짜 openpi Policy(infer 로직 그대로): B=1 우리 경로 == `Policy.infer` 비트 단위,
  3세션 batch == 단독, 소켓 서버 3세션 == 세션별 새 Policy (process 모드), episode 모드는 (seed, 에피소드)에만 의존.
- 모든 파일 컴파일, `--help`(DP 서버·openpi_server·session_server·check_batch·summarize), `run_eval_shared.py`의 교체 명령, `eval_shared.sh DRY_RUN=1`
  (한 손·양손, 공유·옛 경로), server.json 형식이 `eval_mt.sh`와 글자 그대로 같음.

## GPU에서 확인할 것 (vast-isaac, 평가 끝난 뒤 sync)

1. `check_batch.py` (실제 체크포인트): A B=1 공유 경로 == `Policy.infer` (비트 단위가 목표; 노이즈를 jit 밖에서 만들기 때문에 다를 수 있음),
   B batch 3 vs 단독 차이(XLA 커널이 shape마다 다를 수 있음), C B=1..6 추론 시간·메모리.
2. 같은 칸·에피소드 옛(`bench.sh dedicated 2`: 옛 vs 옛 재현성) vs 공유(N=2..5) 행동·성공 비교 (`summarize.py --compare`).
3. 처리량·VRAM: N=2,3,4,5 (32 GB RTX 5090), lockstep vs greedy 1회.
