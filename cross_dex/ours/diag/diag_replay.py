"""Replay with every episode rendered + per-step diagnostics (RSS2027 failure diagnosis, 2026-11-21; main env python).

Runs DexVerse's own ``scripts/record_demos.py --replay_demos`` unchanged (runpy) with two in-process overrides:
  * ReplayDemoDevice.advance: when an episode would be given up (all actions + hold steps without success), the
    episode is finalized instead of RESET, so failed episodes are also rendered into the LeRobot dataset (videos);
  * record_demos.check_success: also logs, every step, the task's REAL success flag, object root positions, robot joint
    positions vs targets and contact-sensor forces -> $DIAG_OUT/ep<k>.npz, summary in $DIAG_OUT/summary.jsonl.
The recorded "success" of the dataset is therefore meaningless here; the real flag is in the npz / summary.

  cd <DexVerse> && PYTHONPATH=source/dexverse DIAG_OUT=<dir> python <this> <record_demos args ...>
"""
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys

import numpy as np

OUT = os.environ["DIAG_OUT"]
os.makedirs(OUT, exist_ok=True)
STATE = {"dev": None, "patched": False, "log": [], "real_step": None, "ep": None}


def _np(x):
    return x.detach().float().cpu().numpy()


def _flush(forced):
    if not STATE["log"]:
        return
    dev, k = STATE["dev"], STATE["ep"]
    keys = STATE["log"][0].keys()
    np.savez_compressed(f"{OUT}/ep{k}.npz", **{kk: np.stack([r[kk] for r in STATE["log"]]) for kk in keys})
    src = dev.episodes[k].get("episode_index") if dev is not None and k < len(dev.episodes) else None
    row = {"episode": k, "source_episode": src, "steps": len(STATE["log"]), "real_success": STATE["real_step"] is not None,
           "real_success_step": STATE["real_step"], "forced": forced}
    open(f"{OUT}/summary.jsonl", "a").write(json.dumps(row) + "\n")
    print("[diag]", row, flush=True)
    STATE["log"], STATE["real_step"] = [], None


def _patch_main_globals(g):
    orig = g["check_success"]
    resolve = g["_resolve_term_callable"]

    def check_success(env, term, count):
        dev = STATE["dev"]
        if STATE["ep"] != dev._ep_idx:
            if STATE["log"]:
                _flush("env_termination")                # previous episode ended by an env termination
            STATE["ep"] = dev._ep_idx
            if not os.path.exists(f"{OUT}/meta.json"):
                r = env.scene["robot"]
                sens = {n: list(getattr(s, "body_names", [])) for n, s in env.scene.sensors.items()
                        if getattr(s.data, "net_forces_w", None) is not None}
                json.dump({"joint_names": list(r.joint_names), "rigid_objects": list(env.scene.rigid_objects.keys()),
                           "contact_sensors": sens, "success_term": str(getattr(term, "func", term)),
                           "success_params": {k: str(v) for k, v in (term.params or {}).items()}},
                          open(f"{OUT}/meta.json", "w"), indent=1)
        real = bool(resolve(env, term)(env, **term.params)[0]) if term is not None else False
        r = env.scene["robot"]
        rec = {"real": np.array(real), "q": _np(r.data.joint_pos[0]), "q_target": _np(r.data.joint_pos_target[0])}
        org = env.scene.env_origins[0]
        for n, o in env.scene.rigid_objects.items():
            rec[f"obj_{n}"] = np.concatenate([_np(o.data.root_pos_w[0] - org), _np(o.data.root_quat_w[0])])
        for n, s in env.scene.sensors.items():
            f = getattr(s.data, "net_forces_w", None)
            if f is not None:
                rec[f"force_{n}"] = _np(f[0].norm(dim=-1))
        STATE["log"].append(rec)
        if real and STATE["real_step"] is None:
            STATE["real_step"] = len(STATE["log"]) - 1
        count, done = orig(env, term, count)
        if done:
            _flush(False)
            return count, True
        if getattr(dev, "_diag_force", False):
            dev._diag_force = False
            _flush(True)
            return count, True
        return count, False

    g["check_success"] = check_success


def _patch_device(mod):
    D = mod.ReplayDemoDevice
    orig_init, orig_advance = D.__init__, D.advance

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        STATE["dev"] = self

    def advance(self):
        if not STATE["patched"]:
            _patch_main_globals(sys._getframe(1).f_globals)
            STATE["patched"] = True
        ep = self.current_episode
        if ep is not None and self._t >= len(ep["actions"]) + self.hold_steps:
            self._diag_force = True                      # finalize (render) instead of RESET
            self._t = len(ep["actions"])                 # keep holding the last action until finalized
        return orig_advance(self)

    D.__init__, D.advance = __init__, advance


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "dexverse.teleop_utils.replay_teleop":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_device(module)

        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Hook())
script = os.path.join(os.getcwd(), "scripts", "record_demos.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
