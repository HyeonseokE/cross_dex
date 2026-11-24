#!/usr/bin/env python3
"""Controller of the multi-hand DP comparison (PC, background loop). Runs on top of the GPU scheduler
(codes/_setup/gpu_sched/sched.py, own daemon: GPU_SCHED_STATE=~/.cache/hscho_share/gpu_sched_xhdp,
GPU_SCHED_SERVERS=<this folder>/sched_servers.json) and only queues jobs / reads results:

  training   9 jobs "xhdp-train-<exp>-<arm>" (A = a6 first, then B = b_noshadow); pinned to their server once started
  snapshots  150k and 200k snapshots (user 2026-11-20) -> shadow only (user 2026-11-20), 10 episodes with distinct layouts, noise seed 0
             (evaluation on p1 GPU2 only; snapshots of other servers are copied there through the PC)
  best       both snapshots evaluated -> highest shadow success, ties -> later step
  final      best -> 30 distinct-layout episodes, one round (user 2026-11-21: evaluate once; was rounds 1-3 until then), shadow only (A and B, user 2026-11-20), videos on
  records    finished run -> results.json / README / settings / meta / INDEX in the run folder, excel rows,
             best + last to HF Dexverse-CSI/shared-action-spaces (private), "ARTIFACT" line for the rollout page

  python3 ctl.py loop            (every 5 min; state ~/.cache/hscho_share/xhdp/state.json, log ctl.log)
  python3 ctl.py once | status
  python3 ctl.py stage <src server> <run_id> <step> <dst server>      (scheduler stage command)
"""
import datetime
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROUNDS = (1,)   # final evaluation rounds (user 2026-11-21: once)

HERE = Path(__file__).resolve().parent
CODE = "/workspace/codes/cross_dex/cross_dex/ours/crosshand_dp"
ST = Path.home() / ".cache/hscho_share/xhdp"
SCHED = Path.home() / ".cache/hscho_share/gpu_sched_xhdp"
SCHED_PY = "/mnt/d/hscho_share/codes/_setup/gpu_sched/sched.py"
SERVERS_CFG = HERE / "sched_servers.json"
TOKEN = "/mnt/d/confidentials/hf_token.json"
RELAY = "Dexverse-CSI/relay-xhdp"
HANDS = ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]
ARMS = ["eh", "e2c", "ea", "xlvla", "joint"]
EXPS = {"a6": HANDS, "b_noshadow": [h for h in HANDS if h != "shadow"]}
EVAL_HANDS = {"a6": ["shadow"], "b_noshadow": ["shadow"]}   # user 2026-11-20 11:05: evaluate shadow only
SNAP_HANDS = ["shadow"]   # user 2026-11-20 11:05: snapshot evaluation = shadow, 10 episodes (all runs)
STEPS = [150000, 200000]   # snapshots that are evaluated (user 2026-11-20 11:40: 150k and 200k only; training still saves every 50k)
LABEL = {"eh": "E-h", "e2c": "E2c", "ea": "E-a", "xlvla": "XL-VLA", "joint": "joint"}
CKR = "/workspace/checkpoints/dexverse/crosshand_dp"
DIRECT = {"vast-isaac2"}
TRAIN_GPUS = {"vast-isaac2": [0, 1], "p1": [2]}
EVAL_GPUS = {"p1": [2]}   # user 2026-11-20 10:00: p1 GPU2 only (not GPU3); vast-isaac2 cannot render (fd limit 1024 -> no GPU foundation)
EVAL_HOST = "p1"
GONE = {"vast-isaac2"}   # returned servers (vast-isaac2 2026-11-21): their evaluated snapshots are already on EVAL_HOST
# per-job usage (EXPERIMENT_RULES 7.1); measured values are written by the measurement step into measured.json
DEFAULT_USE = {"train": {"vram_gb": 12, "ram_gb": 30, "cpu": 12, "disk_gb": 15}, "eval": {"vram_gb": 7, "ram_gb": 12, "cpu": 3, "disk_gb": 1}}


def kst():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))


def log(m):
    line = f"{kst():%m-%d %H:%M} {m}"
    print(line, flush=True)
    with open(ST / "ctl.log", "a") as f:
        f.write(line + "\n")


def sh(server, cmd, timeout=180, stdin=None):
    inner = f"bash -c {shlex.quote(cmd)}" if server in DIRECT else f"docker exec -i hscho_isaac bash -c {shlex.quote(cmd)}"
    try:
        r = subprocess.run(["ssh", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ClearAllForwardings=yes", server, inner],
                           capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def token():
    return json.load(open(TOKEN))["token"] + "\n"


def load_state():
    p = ST / "state.json"
    return json.load(open(p)) if p.exists() else {"runs": {}, "queued": {}}


def save_state(s):
    tmp = ST / "state.json.tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, ST / "state.json")


def use(kind):
    m = ST / "measured.json"
    u = dict(DEFAULT_USE[kind])
    if m.exists():
        u.update(json.load(open(m)).get(kind, {}))
    return u


def sched_job(state_dir, jid):
    for d in ("pending", "running", "done", "failed"):
        p = SCHED / d / f"{jid}.json"
        if p.exists():
            return d, json.load(open(p))
    return None, None


def add_job(job):
    p = SCHED / "pending" / f"{job['id']}.json"
    if any((SCHED / d / f"{job['id']}.json").exists() for d in ("pending", "running", "done", "failed")):
        return False
    json.dump(job, open(p, "w"), indent=1)
    log(f"queued {job['id']}")
    return True


def pin(jid, server):
    for d in ("pending", "running"):
        p = SCHED / d / f"{jid}.json"
        if p.exists():
            j = json.load(open(p))
            if j.get("servers") != [server]:
                j["servers"] = [server]
                json.dump(j, open(p, "w"), indent=1)


def train_job(exp, arm, prio):
    u = use("train")
    m = json.load(open(ST / "measured.json")) if (ST / "measured.json").exists() else {}
    nw, cap = m.get("workers", 12), m.get("max_per_gpu", {"p1": 2, "vast-isaac2": 1})
    jid = f"xhdp-train-{exp}-{arm}"
    return {"id": jid, "cmd": f"bash {CODE}/run_train.sh {arm} {exp} {{gpu}} {nw}", "servers": list(TRAIN_GPUS),
            "gpus": TRAIN_GPUS, **u, "group": "train", "max_per_gpu": cap, "priority": prio, "retries": 3,
            "done": f"ls /workspace/experiments/*/RSS2027/comparison/dexverse-crosshand-dp/graspkettle-*/*_xhdp-{exp.split('_')[0]}-{arm}/logs/trained"}


def ready(server):
    return (ST / f"eval_ready_{server}").exists()


def eval_servers(src, rid, step):
    """Servers that hold the policy: the training server, plus vast-isaac2 once a p1 snapshot was copied there."""
    return [EVAL_HOST]   # snapshots of other servers are copied to EVAL_HOST first (copy_snapshot, job gated on the marker)


def copy_snapshot(src, rid, step, dst=EVAL_HOST):
    """Background copy of one pretrained_model through the PC (ssh pipe, ~1 MB/s; the HF org storage is full,
    2026-11-20). Marker ST/copied_<rid>_<step> when done."""
    mark, busy = ST / f"copied_{rid}_{step}", ST / f"copying_{rid}_{step}"
    if mark.exists() or busy.exists() or src == dst or src in GONE or not ready(dst):
        return
    d = f"{CKR}/{rid}/checkpoints/{step}"
    rd = f"docker exec hscho_isaac tar -C {d} -cf - pretrained_model" if src not in DIRECT else f"tar -C {d} -cf - pretrained_model"
    wr = f"mkdir -p {d} && tar -C {d} -xf - && test -s {d}/pretrained_model/model.safetensors"
    wr = wr if dst in DIRECT else f"docker exec -i hscho_isaac bash -c {shlex.quote(wr)}"
    cmd = (f"ssh -o LogLevel=ERROR -o ClearAllForwardings=yes {src} {shlex.quote(rd)} | "
           f"ssh -o LogLevel=ERROR -o ClearAllForwardings=yes {dst} {shlex.quote(wr)} && touch {mark}; rm -f {busy}")
    busy.touch()
    subprocess.Popen(["setsid", "bash", "-c", cmd], stdout=open(ST / "copy.log", "a"), stderr=subprocess.STDOUT)
    log(f"copy {rid} {step} {src} -> {dst}")


def eval_job(rid, src, exp, arm, hand, kind, tag, eps, seed, prio, video=False):
    """kind snap (tag = step) or final (tag = r<round>, policy = best)."""
    jid = f"xhdp-{kind}-{exp.split('_')[0]}-{arm}-{tag}-{hand}"
    out = f"/workspace/.tmp/xhdp_eval/{rid}/{kind}/{tag}/{hand}"
    step = tag if kind == "snap" else "best"
    pol = f"{CKR}/{rid}/checkpoints/{step}/pretrained_model"
    u = use("eval")
    return {"id": jid, "cmd": f"bash {CODE}/run_cell.sh {arm} {hand} {pol} {eps} {out} {seed} {{gpu}} {'--video' if video else '--no-video'}",
            "servers": eval_servers(src, rid, step), "gpus": EVAL_GPUS, **u, "priority": prio, "retries": 2,
            "out": out, "src": src, "step": step, "meta": {"rid": rid, "exp": exp, "arm": arm, "hand": hand, "kind": kind, "tag": tag}}


# ---------------------------------------------------------------- stage: snapshot to another server via the HF relay
def stage(src, rid, step, dst):
    path = f"{CKR}/{rid}/checkpoints/{step}/pretrained_model"
    rc, _ = sh(dst, f"test -s {path}/model.safetensors")
    if rc == 0:
        return 0
    up = (f"read T; HF_TOKEN=$T HF_HOME=/workspace/.cache/huggingface /workspace/envs/dexverse/lerobot/bin/python -c "
          + shlex.quote(f"from huggingface_hub import HfApi; a=HfApi(); a.upload_folder(repo_id='{RELAY}', repo_type='dataset', "
                        f"folder_path='{path}', path_in_repo='ckpt/{rid}/{step}', commit_message='{rid} {step}')"))
    rc, out = sh(src, up, timeout=3600, stdin=token())
    if rc != 0:
        print(f"upload failed {rc} {out[-300:]}", file=sys.stderr)
        return 1
    down = (f"read T; mkdir -p {path}; HF_TOKEN=$T HF_HOME=/workspace/.cache/huggingface /workspace/envs/dexverse/dexlatent/bin/python -c "
            + shlex.quote(f"from huggingface_hub import snapshot_download; import shutil, glob, os; "
                          f"d=snapshot_download('{RELAY}', repo_type='dataset', allow_patterns=['ckpt/{rid}/{step}/*']); "
                          f"[shutil.copy(f, '{path}') for f in glob.glob(d + '/ckpt/{rid}/{step}/*')]"))
    rc, out = sh(dst, down + f" && test -s {path}/model.safetensors", timeout=3600, stdin=token())
    if rc != 0:
        print(f"download failed {rc} {out[-300:]}", file=sys.stderr)
    return rc


# ---------------------------------------------------------------- one pass
def run_info(server, exp, arm):
    rc, out = sh(server, f"cat {CKR}/runs/{exp}_{arm}.rid 2>/dev/null; "
                         f"for d in {CKR}/$(cat {CKR}/runs/{exp}_{arm}.rid 2>/dev/null)/checkpoints/[0-9]*; do "
                         f"[ -s $d/pretrained_model/model.safetensors ] && [ -s $d/training_state/training_step.json ] && basename $d; done; true")
    lines = out.split()
    if rc != 0 or not lines:
        return None, []
    return lines[0], sorted(int(x) for x in lines[1:] if x.isdigit())


def metrics(server, out):
    rc, o = sh(server, f"cat {out}/eval/metrics.json 2>/dev/null")
    try:
        return json.loads(o)
    except Exception:
        return None


def once():
    s = load_state()
    for exp_i, exp in enumerate(EXPS):
        for arm_i, arm in enumerate(ARMS):
            if arm == "joint" and exp != "a6":            # joint baseline: experiment A only (no shared space for an unseen hand)
                continue
            key = f"{exp}/{arm}"
            r = s["runs"].setdefault(key, {"exp": exp, "arm": arm, "snaps": {}, "final": {}})
            jid = f"xhdp-train-{exp}-{arm}"
            extra = json.load(open(ST / "measure_runs.json")) if (ST / "measure_runs.json").exists() else []
            if not (ST / "measured_done").exists() and key not in s.get("measure_runs", []) + extra:
                continue                                   # measurement first (only the measurement runs are queued)
            add_job(train_job(exp, arm, prio=100 - 10 * exp_i - arm_i))
            st, j = sched_job(SCHED, jid)
            if j and j.get("server"):
                r["server"] = j["server"]
                pin(jid, j["server"])
            if not r.get("server"):
                continue
            if r["server"] in GONE:
                rid = r.get("rid")
                steps = [x for x in STEPS if rid and (ST / f"copied_{rid}_{x:06d}").exists()]
            else:
                rid, steps = run_info(r["server"], exp, arm)
            if not rid:
                continue
            r["rid"] = rid
            for step in steps:                             # snapshot evaluations
                if int(step) not in STEPS:
                    continue
                tag = f"{step:06d}"
                copy_snapshot(r["server"], rid, tag)
                for h in SNAP_HANDS:
                    cell = r["snaps"].setdefault(tag, {}).get(h)
                    if cell and cell.get("done"):
                        continue
                    if r["server"] != EVAL_HOST and not (ST / f"copied_{rid}_{tag}").exists():
                        continue   # wait for the copy to EVAL_HOST
                    job = eval_job(rid, r["server"], exp, arm, h, "snap", tag, "unique10", 0, prio=200 + step // 50000)
                    add_job(job)
                    jst, jj = sched_job(SCHED, job["id"])
                    if jst == "done":
                        m = metrics(jj["server"], job["out"])
                        if m:
                            r["snaps"][tag][h] = {"done": True, "success": sum(e["success"] for e in m["episodes"]),
                                                  "n": m["num_episodes"], "server": jj["server"]}
                    elif jst == "failed":
                        r["snaps"][tag][h] = {"done": False, "failed": True}
            full = [f"{x:06d}" for x in STEPS]
            if not r.get("best") and all(all(r["snaps"].get(t, {}).get(h, {}).get("done") for h in SNAP_HANDS) for t in full):
                tot = {t: sum(r["snaps"][t][h]["success"] for h in SNAP_HANDS) for t in full}
                best = max(full, key=lambda t: (tot[t], t))
                bsrv = EVAL_HOST if r["server"] in GONE else r["server"]
                rc, _ = sh(bsrv, f"cd {CKR}/{rid}/checkpoints && rm -rf best && cp -r {best} best")
                if rc == 0 and bsrv != r["server"]:
                    (ST / f"copied_{rid}_best").touch()
                if rc == 0:
                    r["best"], r["snap_totals"] = best, tot
                    log(f"{key} best {best} ({tot})")
            if r.get("best"):
                copy_snapshot(r["server"], rid, "best")
                for h in EVAL_HANDS[exp]:
                    for rnd in ROUNDS:
                        tag = f"r{rnd}"
                        cell = r["final"].setdefault(tag, {}).get(h)
                        if cell and cell.get("done"):
                            continue
                        if r["server"] != EVAL_HOST and not (ST / f"copied_{rid}_best").exists():
                            continue
                        job = eval_job(rid, r["server"], exp, arm, h, "final", tag, "unique30", rnd, prio=300 - rnd, video=rnd == 1)
                        add_job(job)
                        jst, jj = sched_job(SCHED, job["id"])
                        if jst == "done":
                            m = metrics(jj["server"], job["out"])
                            if m:
                                r["final"][tag][h] = {"done": True, "success": sum(e["success"] for e in m["episodes"]),
                                                      "n": m["num_episodes"], "server": jj["server"], "out": job["out"]}
                if not r.get("finished") and all(r["final"].get(f"r{k}", {}).get(h, {}).get("done") for h in EVAL_HANDS[exp] for k in ROUNDS):
                    finish_run(r)
    # eval jobs may move to vast-isaac2 once it is ready
    for p in (SCHED / "pending").glob("xhdp-*.json"):
        j = json.load(open(p))
        if "meta" in j and set(j["servers"]) != set(eval_servers(j["src"], j["meta"]["rid"], j["step"])):
            j["servers"] = eval_servers(j["src"], j["meta"]["rid"], j["step"])
            json.dump(j, open(p, "w"), indent=1)
    save_state(s)


# ---------------------------------------------------------------- records when a run is finished
def finish_run(r):
    exp, arm, rid, srv = r["exp"], r["arm"], r["rid"], r["server"]
    res = {"run_id": rid, "arm": arm, "label": LABEL[arm], "exp": exp, "best": r["best"], "snap_totals": r["snap_totals"],
           "snapshots": r["snaps"], "final": {}}
    for h in EVAL_HANDS[exp]:
        v = [r["final"][f"r{k}"][h]["success"] / r["final"][f"r{k}"][h]["n"] for k in ROUNDS]
        res["final"][h] = {"rounds": v, "mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0}
    if exp == "b_noshadow":   # training hands of B: the snapshot evaluation of the best step (user, 2026-11-20)
        res["train_hands_snapshot"] = {h: r["snaps"][r["best"]][h]["success"] / r["snaps"][r["best"]][h]["n"] for h in SNAP_HANDS}
    if (ST / "measured.json").exists():
        res["measured"] = json.load(open(ST / "measured.json"))
    json.dump(res, open(ST / f"result_{exp}_{arm}.json", "w"), indent=1)
    if r.get("records_done") or srv in GONE:   # returned server: run folder only in central experiments/<srv>/, main agent writes it
        rc = rc2 = 0
        if srv in GONE and not r.get("records_done"):
            log(f"records of {rid}: server {srv} returned, main agent writes them in central experiments/{srv}/")
    else:
        rc, run_dir = sh(srv, f"ls -d /workspace/experiments/*/RSS2027/comparison/dexverse-crosshand-dp/graspkettle-*/{rid}")
        run_dir = run_dir.strip()
        rc, _ = sh(srv, f"mkdir -p {run_dir}/artifacts && cat > {run_dir}/artifacts/results.json", stdin=json.dumps(res, indent=1))
        rc2, _ = sh(srv, f"python3 {CODE}/records.py {run_dir} {CKR}/{rid}", timeout=600)
    r["records_done"] = rc == 0 and rc2 == 0
    if not r.get("excel_done"):   # finish_run is retried until HF succeeds; add the excel rows once
        excel_rows(res); r["excel_done"] = True
    rc3 = 0 if r.get("hf_done") else hf_upload(EVAL_HOST if srv in GONE else srv, rid, exp, arm, r["best"])
    r["hf_done"] = rc3 == 0
    r["finished"] = rc == 0 and rc2 == 0 and rc3 == 0
    log(f"FINISHED {exp}/{arm} {rid}: " + ", ".join(f"{h} {v['mean']:.2f}±{v['std']:.2f}" for h, v in res["final"].items())
        + f" (records rc {rc2}, hf rc {rc3})")
    print(f"ARTIFACT {exp} {arm} {rid} — rollout page needs the main agent", flush=True)


def hf_upload(srv, rid, exp, arm, best):
    cmd = (f"read T; HF_TOKEN=$T HF_HOME=/workspace/.cache/huggingface /workspace/envs/dexverse/lerobot/bin/python -c "
           + shlex.quote(f"from huggingface_hub import HfApi; a=HfApi(); r='Dexverse-CSI/shared-action-spaces'; "
                         f"assert a.repo_info(r).private; "
                         f"[a.upload_folder(repo_id=r, folder_path=f'{CKR}/{rid}/checkpoints/{{s}}/pretrained_model', "
                         f"path_in_repo=f'policy/dexverse/crosshand-dp-graspkettle/{exp}/{arm}/{{n}}', commit_message=f'{rid} {{n}}') "
                         f"for s, n in (('{best}', 'best_{best}'), ('200000', 'last_200000'))]"))
    rc, out = sh(srv, cmd, timeout=7200, stdin=token())
    return rc


def excel_rows(res):
    rows = []
    exp_name = "다중손DP_A (6손)" if res["exp"] == "a6" else "다중손DP_B (shadow 제외)"
    for h, v in res["final"].items():
        for k, x in enumerate(v["rounds"], 1):
            rows.append([exp_name, "DexVerse", "GraspKettle", h, f"DP_{res['label']}", round(x * 30), 30,
                         f"{kst():%Y-%m-%d}", res["run_id"], k])
    pend = ST / "excel_pending.json"
    allrows = (json.load(open(pend)) if pend.exists() else []) + rows
    json.dump(allrows, open(pend, "w"), indent=1)
    flush_excel()


def flush_excel():
    pend = ST / "excel_pending.json"
    if not pend.exists():
        return
    desk = Path("/mnt/c/Users/hscho_windows/Desktop")
    latest, orig = desk / "Cross-Dex_RSS2027_results_최신본.xlsx", desk / "Cross-Dex_RSS2027_results.xlsx"
    if any(desk.glob("~$Cross-Dex_RSS2027_results*")):
        log("excel open in Excel: rows kept in excel_pending.json")
        return
    rows = json.load(open(pend))
    spec = ST / "excel_rows.json"
    json.dump({"rows": rows}, open(spec, "w"))
    subprocess.run(["cp", str(latest), str(orig)])
    r = subprocess.run(["uv", "run", "-q", "--with", "openpyxl", "python", str(HERE / "xlsx_rows.py"), str(latest), str(spec)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        pend.unlink()
        log(f"excel: {len(rows)} rows added to 결과_원본 ({latest.name})")
    else:
        log(f"excel failed: {r.stderr[-300:]}")


def status():
    s = load_state()
    for k, r in s["runs"].items():
        snaps = {t: f"{sum(c.get('success', 0) for c in v.values())}/{sum(c.get('n', 0) for c in v.values())}" for t, v in r["snaps"].items()}
        print(k, r.get("server"), r.get("rid"), "snaps", snaps, "best", r.get("best"), "final",
              {t: {h: c.get("success") for h, c in v.items()} for t, v in r["final"].items()}, "finished" if r.get("finished") else "")


def main():
    ST.mkdir(parents=True, exist_ok=True)
    a = sys.argv[1:]
    if a[0] == "stage":
        sys.exit(stage(*a[1:5]))
    if a[0] == "status":
        return status()
    if a[0] == "once":
        return once()
    if a[0] == "loop":
        log("ctl loop start")
        while True:
            try:
                once()
                flush_excel()
            except Exception as e:
                log(f"ctl error: {e!r}")
            s = load_state()
            if len(s["runs"]) == 9 and all(r.get("finished") for r in s["runs"].values()):
                log("ALL FINISHED")
                print("ALL FINISHED", flush=True)
                break
            time.sleep(300)


if __name__ == "__main__":
    main()
