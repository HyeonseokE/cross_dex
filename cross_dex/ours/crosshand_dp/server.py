#!/usr/bin/env python
"""Policy server of the multi-hand DPs for the unchanged common evaluator (eval_lerobot.py, through run_eval.py here).
Same protocol as its policy_server.py: ("reset", None) -> "ok"; ("act", obs) -> float32 action bytes.

--ckpt = folder with server.json:
  {"arm": "<space>" | "joint", "hand": h, "task": t, "policy_ckpt": <pretrained_model>, "latent_ckpt": <space model.pt>
   (joint: the XL-VLA checkpoint, hand models only), "wrist": <wrist_calib_<task>.json>, "episodes": [pickle episodes in
   evaluator order], "noise_seed": null | int, "hand_onehot": [hand order] (joint only), "log_dir": null}
space arm: dexverse_xlvla.policy.LatentPolicyServer unchanged (state [wrist 9 | E_h(q)], action -> IK wrist, D_h(z)).
joint arm: state [wrist 9 | fingers normalised, zero-padded to 22 | hand one-hot], action fingers = first n_h values.
noise_seed: torch seed noise_seed * 100000 + <pickle episode> at every reset (as shared_eval batch_server.episode_seed),
so one episode's diffusion noise does not depend on the other episodes of the cell; null = unseeded (as before).
"""
import argparse
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from dexverse_xlvla.policy import LatentPolicyServer

PAD = 22


class Server(LatentPolicyServer):
    tm = None; _d_state = None; _d_act = (); _checked = False   # wrist-offset caches / timing (set at reset)
    def __init__(self, cfg):
        super().__init__(dict(cfg, mode="latent_dp"))
        self.joint = cfg["arm"] == "joint"
        if self.joint:
            order = cfg["hand_onehot"]
            self.onehot = np.asarray([float(h == self.hand) for h in order], np.float32)
        self.episodes, self.seed = cfg.get("episodes"), cfg.get("noise_seed")
        self.fix = None
        if cfg.get("residual_fix"):   # self-residual module (residual_fix.py): fingers = D_h(z) + f_h(z)
            from cross_dex.ours.diag.decoder_fix import mlp
            F = torch.load(cfg["residual_fix"], map_location="cpu", weights_only=False)
            assert F["latent_ckpt"] == cfg["latent_ckpt"], (F["latent_ckpt"], cfg["latent_ckpt"])
            self.fix = mlp(F["dof"][self.hand]); self.fix.load_state_dict(F["state_dicts"][self.hand]); self.fix.eval()
        self.wo = None
        if cfg.get("wrist_f"):        # wrist offset model at inference (wrist_pairs/f_wrist.py PolicyWristOffset), space arms only
            from cross_dex.ours.wrist_pairs.f_wrist import PolicyWristOffset
            assert not self.joint
            fixes = {}
            if cfg.get("residual_fix"):
                for h in cfg["wrist_f"]["sources"]:
                    if h in F["state_dicts"]:
                        fixes[h] = mlp(F["dof"][h]); fixes[h].load_state_dict(F["state_dicts"][h]); fixes[h].eval()
            self.wo = PolicyWristOffset(cfg["wrist_f"], self.tr, self.hand, fixes)

    def reset(self):
        super().reset()
        if getattr(self, "tm", None) and self.tm["steps"]:
            print("[server] timing s/episode " + " ".join(f"{k}={v:.1f}" for k, v in self.tm.items()), file=sys.stderr, flush=True)
        self.tm = {"policy": 0.0, "wo_state": 0.0, "wo_act": 0.0, "wrist_ik": 0.0, "steps": 0}; self._d_state, self._d_act, self._checked = None, [], False
        if self.seed is not None:
            ep = self.episodes[self.ep] if self.episodes else self.ep
            torch.manual_seed(int(self.seed) * 100000 + int(ep))

    def _fingers(self, q):
        return self.m.angles_to_normalized(torch.tensor(q[None, :len(self.names)][:, self.idx], dtype=torch.float32)).clamp(-1, 1)

    @staticmethod
    def _mat(v):
        from dexverse_xlvla.wrist import from_rot6d
        t = np.eye(4); t[:3, :3] = from_rot6d(np.asarray(v[3:9], np.float64)); t[:3, 3] = v[:3]
        return t

    @staticmethod
    def _vec(t):
        return np.concatenate([t[:3, 3], t[:3, 0], t[:3, 1]])

    def _encode_state(self, q):
        if not self.joint:
            s = super()._encode_state(q)
            if self.wo is not None:   # observed wrist -> the wrist the policy's training hands would have
                n = len(self.names); ang = q[None, :n][:, self.idx]
                # the policy reads only the last n_obs_steps observations before it draws a new chunk: the offset of the
                # other steps is never used, so it is not recomputed there (same policy input, ~4x fewer FK calls)
                if self._d_state is None or len(self.policy._queues["action"]) < self.policy.config.n_obs_steps:
                    t0 = time.perf_counter()
                    self._d_state = self.wo.delta(torch.tensor(s[None, 9:], dtype=torch.float32), list(self.m.dof_joints), ang)
                    self.tm["wo_state"] += time.perf_counter() - t0
                s = s.copy(); s[:9] = self._vec(self._mat(s[:9]) @ np.linalg.inv(self._d_state))
            return s
        f = self._fingers(q)[0].numpy()
        pad = np.zeros(PAD, np.float32)
        pad[:len(f)] = f
        return np.concatenate([self.wrist.vec_from_joints(q[None, :6])[0], pad, self.onehot]).astype(np.float32)

    def act(self, message):
        if not self.joint and self.fix is None and self.wo is None:
            return super().act(message)
        if not self.joint:
            return self._act_fixed(message)
        batch, q = {}, None
        for key, (shape, dtype, data) in message.items():
            v = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            if key == "observation.state":
                q = v.astype(np.float64)
                v = self._encode_state(q)
            t = torch.from_numpy(v)
            batch[key] = t.permute(2, 0, 1).float() / 255.0 if t.ndim == 3 else t.float()
        new_chunk = len(self.policy._queues["action"]) == 0
        with torch.inference_mode():
            out = self.post(self.policy.select_action(self.pre(batch)))[0].cpu().numpy().astype(np.float64)
        act = np.zeros(len(self.names))
        act[:6] = self.wrist.actions_from_vec(out[None, :9], q_init=q[:6])[0]
        qn = np.clip(out[9:9 + len(self.idx)], -1, 1)
        lo, hi = self.m._lower.numpy(), self.m._upper.numpy()
        act[self.idx] = (qn + 1) * 0.5 * (hi - lo) + lo
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        return act.astype(np.float32)

    def _act_fixed(self, message):   # = LatentPolicyServer.act (latent branch) with the decoder D_h + f_h
        from dexverse_xlvla.eval_latent import dec
        batch, q = {}, None
        for key, (shape, dtype, data) in message.items():
            v = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            if key == "observation.state":
                q = v.astype(np.float64)
                v = self._encode_state(q)
            t = torch.from_numpy(v)
            batch[key] = t.permute(2, 0, 1).float() / 255.0 if t.ndim == 3 else t.float()
        new_chunk = len(self.policy._queues["action"]) == 0
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.post(self.policy.select_action(self.pre(batch)))[0].cpu().numpy().astype(np.float64)
            if self.wo is not None and new_chunk:   # wrist offsets of the whole chunk in one batch
                rest = [self.post(x)[0].cpu().numpy().astype(np.float64) for x in self.policy._queues["action"]]
        self.tm["policy"] += time.perf_counter() - t0
        if self.wo is not None and new_chunk:
            t0 = time.perf_counter()
            with torch.no_grad():
                Z = torch.tensor(np.stack([out] + rest)[:, 9:], dtype=torch.float32); Q = dec(self.tr, self.hand, Z)
                if self.fix is not None:
                    Q = Q + self.fix(Z)
                Q = Q.clamp(-1, 1)
            ang = ((Q + 1) * 0.5 * (self.m._upper - self.m._lower) + self.m._lower).double().numpy()
            self._d_act = list(self.wo.delta_batch(Z, list(self.m.dof_joints), ang)); self.tm["wo_act"] += time.perf_counter() - t0
            if not self._checked:   # once per episode: the batched offset equals the per-step one
                d1 = self.wo.delta(Z[:1], list(self.m.dof_joints), ang[:1]); self._checked = True
                assert np.abs(d1 - self._d_act[0]).max() < 1e-5, np.abs(d1 - self._d_act[0]).max()
        act = np.zeros(len(self.names))
        with torch.no_grad():
            z = torch.tensor(out[None, 9:], dtype=torch.float32)
            qn = dec(self.tr, self.hand, z)
            if self.fix is not None:
                qn = qn + self.fix(z)
            qn = qn[0].clamp(-1, 1)
        lo, hi = self.m._lower, self.m._upper
        act[self.idx] = ((qn + 1) * 0.5 * (hi - lo) + lo).numpy()
        w = out[None, :9]
        if self.wo is not None:
            d = self._d_act.pop(0)
            w = self._vec(self._mat(out[:9]) @ d)[None]; self.log.setdefault("wrist_delta", []).append(d)
        t0 = time.perf_counter(); act[:6] = self.wrist.actions_from_vec(w, q_init=q[:6])[0]; self.tm["wrist_ik"] += time.perf_counter() - t0; self.tm["steps"] += 1
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        self.log.setdefault("q", []).append(q)   # observed joints (failure analysis, diag/zs_fail.py)
        return act.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--authkey", required=True)
    a = ap.parse_args()
    policy = Server(json.loads((Path(a.ckpt) / "server.json").read_text()))
    with Listener(("127.0.0.1", a.port), authkey=a.authkey.encode()) as listener, listener.accept() as conn:
        while True:
            kind, message = conn.recv()
            if kind == "close":
                policy.close()
                break
            if kind == "reset":
                policy.reset()
                conn.send("ok")
                continue
            conn.send(np.asarray(policy.act(message), dtype=np.float32).tobytes())
