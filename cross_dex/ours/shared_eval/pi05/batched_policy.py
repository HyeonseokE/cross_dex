"""Batched inference with an openpi Policy (JAX model), one row per evaluation session, each row with its session's noise.

openpi Policy.infer (src/openpi/policies/policy.py) does, per call:
  inputs = input_transform(obs); add a batch axis of 1
  self._rng, sample_rng = jax.random.split(self._rng)          (self._rng = jax.random.key(0) at creation)
  actions = sample_actions(sample_rng, Observation.from_dict(inputs))   -> noise = jax.random.normal(sample_rng, (1, H, D))
  outputs = output_transform({"state": inputs["state"][0], "actions": actions[0]})
infer_batch() does the same for B observations at once: the input/output transforms run per row (the policy's own
transform objects), rows are stacked on the batch axis (each leaf converted with jnp.asarray first, as infer does), and the
model gets an explicit noise array whose row i is jax.random.normal(sample_rng_i, (1, H, D)) drawn from session i's key.
SessionKey keeps that key per session:
  process mode: key(0) when the session connects, split once per chunk -> the noise sequence of the baseline's own
                openpi process (pi05/serve.py, one per cell, created with the default key(0)).
  episode mode: key(episode_seed(seed, episode)) at every episode reset -> independent of the cell's history.
Nothing here depends on which sessions share a batch; only the model's batched arithmetic can differ from B=1
(XLA kernels per shape; to be measured on GPU, pi05/check_batch.py).
"""
from __future__ import annotations

import numpy as np


class SessionKey:
    def __init__(self, jax, mode: str = "process", seed: int = 0):
        if mode not in ("process", "episode"):
            raise ValueError(mode)
        self.jax, self.mode, self.seed = jax, mode, seed
        self.key = jax.random.key(0)  # = openpi Policy(rng=None)

    def reset(self, episode):
        if self.mode == "episode":
            from batch_server import episode_seed, fresh_seed
            self.key = self.jax.random.key(episode_seed(self.seed, episode) if episode is not None else fresh_seed())

    def next(self):
        """= Policy.infer: self._rng, sample_rng = jax.random.split(self._rng)"""
        self.key, sample = self.jax.random.split(self.key)
        return sample


class BatchedPolicy:
    def __init__(self, policy, observation_from_dict, jax, jnp):
        """policy: openpi Policy (JAX); observation_from_dict: openpi.models.model.Observation.from_dict."""
        if getattr(policy, "_is_pytorch_model", False):
            raise NotImplementedError("PyTorch openpi checkpoints: not supported (the trained pi0.5 checkpoints are JAX)")
        self.policy, self.from_dict, self.jax, self.jnp = policy, observation_from_dict, jax, jnp
        model = policy._model
        self.shape = (1, model.action_horizon, model.action_dim)
        self._normal = jax.jit(lambda key: jax.random.normal(key, self.shape))  # same draw as inside sample_actions

    def noise(self, sample_key):
        return self._normal(sample_key)

    def infer_batch(self, observations, sample_keys):
        """observations: list of obs dicts (as sent to Policy.infer); sample_keys: one key per row. -> list of outputs."""
        jax, jnp, p = self.jax, self.jnp, self.policy
        inputs = [p._input_transform(jax.tree.map(lambda x: x, obs)) for obs in observations]
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inputs)
        noise = jnp.concatenate([self.noise(k) for k in sample_keys])
        actions = p._sample_actions(sample_keys[0], self.from_dict(batch), **dict(p._sample_kwargs), noise=noise)
        outputs = {"state": batch["state"], "actions": actions}
        return [p._output_transform(jax.tree.map(lambda x: np.asarray(x[i, ...]), outputs)) for i in range(len(inputs))]
