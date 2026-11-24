#!/usr/bin/env python3
"""Run files of a finished multi-hand DP run (EXPERIMENT_RULES 2-3), from what the run actually used/produced:
config.yaml, logs/train.log, artifacts/results.json (ctl.py), the checkpoint's train_config.json.
Writes metrics.csv, settings.md, README.md, meta.json (end, status done), one INDEX.csv line, loss curve (if matplotlib).

  python3 records.py <run_dir> <checkpoint dir>
"""
import csv
import datetime
import json
import os
import re
import subprocess
import sys

run, ck = sys.argv[1:3]
res = json.load(open(f"{run}/artifacts/results.json"))
cfg = dict(l.split(": ", 1) for l in open(f"{run}/config.yaml").read().splitlines() if ": " in l)
meta = json.load(open(f"{run}/meta.json"))
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")

# metrics.csv: training loss from the LeRobot log + evaluation results
rows = []
frames = json.load(open(f"{cfg['dataset_root']}/meta/info.json"))["total_frames"]
for line in open(f"{run}/logs/train.log", errors="ignore"):
    m = re.search(r" epch:([0-9.]+) loss:([0-9.e+-]+)", line)   # step = epoch x steps per epoch (the log rounds step to 1K)
    if m:
        rows.append((round(float(m.group(1)) * frames / 64), "train", "loss", float(m.group(2))))
for t, cells in res["snapshots"].items():
    for h, c in cells.items():
        rows.append((int(t), "snapshot_eval", f"success_{h}", c["success"] / c["n"]))
for h, v in res["final"].items():
    for k, x in enumerate(v["rounds"], 1):
        rows.append((int(res["best"]), f"final_r{k}", f"success_{h}", x))
with open(f"{run}/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "split", "metric", "value"])
    w.writerows(rows)

tc = {}
try:
    tc = json.load(open(f"{ck}/checkpoints/{res['best']}/pretrained_model/train_config.json"))
except Exception:
    pass
pol = tc.get("policy", {})
ms = res.get("measured", {})
fin = "\n".join(f"| {h} | {' / '.join(f'{x:.2f}' for x in v['rounds'])} | {v['mean']:.2f} ± {v['std']:.2f} |" for h, v in res["final"].items())
snap = "\n".join(f"| {t} | {' | '.join(str(c['success']) + '/' + str(c['n']) for c in cells.values())} | {res['snap_totals'][t]} |"
                 for t, cells in sorted(res["snapshots"].items()) if t in res["snap_totals"])   # only the steps used for best (old 50k cells skipped)
hands_snap = list(res["snapshots"][res["best"]].keys())
open(f"{run}/settings.md", "w").write(f"""# settings — {res['run_id']}

## 태스크·데이터
- 태스크 Dexverse-GraspKettle-v0 (떠 있는 오른손), 텔레옵 데모 손당 50개 (`{cfg.get('source_root')}`, 목표 구 렌더링본)
- 학습 손: {cfg.get('hands')} (실험 {res['exp']}), 데이터셋 `{cfg.get('dataset_root')}`, 분할 없음 (전부 학습)

## 입력 (Input)
- 이미지: third_person·eye_in_hand 2대, 480x640 → 데이터 로더에서 240x320 축소, crop 216x288 (카메라별 RGB 인코더)
- 상태: [손목 정준 자세 9 (위치 3 + rot6d 6) | {'손가락 관절 정규화 22 (0 패딩) + 손 one-hot 6' if res['arm'] == 'joint' else '공간 z (E_h 사후 평균)'}], 관측 길이 {pol.get('n_obs_steps', '기록 없음')}
- 정규화: LeRobot 기본 (상태·행동 min-max, 이미지 mean-std)

## 출력 (Output)
- 행동: [손목 정준 자세 9 | {'손가락 관절 정규화 22' if res['arm'] == 'joint' else 'z'}] 절대값, 실행 때 손목 = IK, 손가락 = {'앞 n_h 값' if res['arm'] == 'joint' else 'D_h(z)'}
- 청크 {pol.get('horizon', '기록 없음')}, 실행 {pol.get('n_action_steps', '기록 없음')} step

## 모델
- LeRobot Diffusion Policy (ResNet18 처음부터, U-Net), 공간 체크포인트 `{cfg.get('space_ckpt')}` ({res['label']})

## 학습 하이퍼파라미터
- Adam lr {pol.get('optimizer_lr', '1e-4')}, cosine warmup {pol.get('scheduler_warmup_steps', 500)}, batch 64, {cfg.get('steps')} step, 저장 {cfg.get('save_freq')} step마다, seed 42, fp32, early stop 없음
- 데이터 로더 워커 {cfg.get('num_workers')}, 학습 코드 commit {meta.get('git_commit')} (dirty)
- 자원 측정·병렬 수 (EXPERIMENT_RULES 7.1): {json.dumps(ms, ensure_ascii=False) if ms else '기록 없음'}

## 추론(평가) 하이퍼파라미터
- 공통 평가기 eval_lerobot.py (ours/crosshand_dp/run_eval.py로 정책 서버만 교체), DDPM {pol.get('num_inference_steps') or 100} step, 최대 3000 step, 성공 = 태스크 성공 항
- 스냅숏 선택: 저장본 4개 × 학습 손마다 서로 다른 초기 배치 10 에피소드, 잡음 시드 0, best = 학습 손 성공 합 최고 (동률이면 뒤 step) → **{res['best']}**
- 최종: best, 칸마다 서로 다른 초기 배치 30 에피소드 × 3회 (잡음 시드 = 회차), 평가 손 {list(res['final'])}

| step | {' | '.join(hands_snap)} | 합 |
|---|{'---|' * len(hands_snap)}---|
{snap}

| 평가 손 | 회차 1 / 2 / 3 | 평균 ± 표준편차 |
|---|---|---|
{fin}
""")

mean_all = sum(v["mean"] for v in res["final"].values()) / len(res["final"])
open(f"{run}/README.md", "w").write(f"""# {res['run_id']}

## 목적
다중손·단일 태스크 DP(GraspKettle)로 공유 행동 공간 {res['label']}의 정책 성능 비교 ({'실험 A: 6손 함께' if res['exp'] == 'a6' else '실험 B: shadow를 빼고 학습 → shadow 처음 보는 손'}).

## 변경점
같은 묶음의 다른 run과 행동 공간만 다르다 ({res['label']}). 데이터·DP 설정·평가 동일.

## 결과
best {res['best']}, 최종 성공률 (30회 × 3): {', '.join(f"{h} {v['mean']:.2f} ± {v['std']:.2f}" for h, v in res['final'].items())}{' / 학습 손(스냅숏 10회): ' + ', '.join(f'{h} {x:.1f}' for h, x in res.get('train_hands_snapshot', {}).items()) if res.get('train_hands_snapshot') else ''}.
체크포인트: HF `Dexverse-CSI/shared-action-spaces` (private) `policy/dexverse/crosshand-dp-graspkettle/{res['exp']}/{res['arm']}/{{best_{res['best']},last_200000}}`.

## 결론
평가 손 평균 {mean_all:.2f}. 공간 간 비교는 묶음 표에서 함께 본다.
""")
meta.update(end_time=now, status="done")
json.dump(meta, open(f"{run}/meta.json", "w"), indent=1)
idx = os.path.join(os.path.dirname(run), "INDEX.csv")
new = not os.path.exists(idx)
with open(idx, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["run_id", "status", "main_metric", "main_value", "summary"])
    w.writerow([res["run_id"], "done", "final_success_mean", f"{mean_all:.3f}",
                f"{res['label']} {res['exp']} best {res['best']}: " + "; ".join(f"{h} {v['mean']:.2f}±{v['std']:.2f}" for h, v in res["final"].items())])
subprocess.run(["/workspace/envs/dexverse/lerobot/bin/python", "-c",
                f"import sys; sys.path.insert(0, '/workspace/codes/cross_dex'); sys.argv=['p', '{run}']; "
                "from cross_dex.ours.plot_curves import main; main()"], check=False)
print("[records] done", run)
