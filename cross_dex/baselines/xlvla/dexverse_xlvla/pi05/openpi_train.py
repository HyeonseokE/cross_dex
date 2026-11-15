"""pi0.5 on the XL-VLA latent action space, built on openpi WITHOUT editing it (config, transforms and weight loader live
here and are passed to openpi's own train / norm-stats code).

Data: the make_dataset LeRobot v3.0 set (<out>/all) read by dexverse_xlvla.pi05.v3_dataset.V3Dataset, swapped in for
openpi's v2.1-only lerobot loader (openpi.training.data_loader.create_torch_dataset is replaced at run time, openpi files
untouched). --repo_id is only the name of the norm-stats asset folder; --root is the dataset.
Policy I/O:
  images  : third_person -> base_0_rgb, eye_in_hand -> left_wrist_0_rgb (right wrist slot masked), as openpi LIBERO / DexVerse pi0.5
  state   : [canonical wrist 9 | z 32] -> pi0.5 discrete state tokens in the prompt (discrete_state_input=True: XL-VLA
            "replace pi0's state tokens with latent tokens")
  prompt  : task sentence (no hand identity, as XL-VLA)
  actions : chunk of [canonical wrist 9 | z 32]; decoded per hand by the frozen VAE decoders at evaluation
pi0.5 base weights are loaded except the action in/out projections, whose shape depends on action_dim (32 in the base
model, 41 here): those start from init (ShapeTolerantWeightLoader).

  python -m dexverse_xlvla.pi05.openpi_train norm  --root <v3 dir> --repo_id <id> --run <name>
  python -m dexverse_xlvla.pi05.openpi_train train --root <v3 dir> --repo_id <id> --run <name> [--steps ... --batch ...]
Env: openpi venv; OPENPI_DATA_HOME (base weights cache), XLVLA_PI05_OUT (assets+checkpoints).
"""
import argparse, dataclasses, logging, os, pathlib, sys

import einops
import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def _img(x) -> np.ndarray:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        x = (255 * x).astype(np.uint8)
    if x.shape[0] == 3:
        x = einops.rearrange(x, "c h w -> h w c")
    return x


@dataclasses.dataclass(frozen=True)
class LatentInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _img(data["image"])
        out = {"state": np.asarray(data["state"], np.float32),
               "image": {"base_0_rgb": base, "left_wrist_0_rgb": _img(data["wrist_image"]), "right_wrist_0_rgb": np.zeros_like(base)},
               "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                              "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_}}
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class LatentOutputs(_transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class LatentDataConfig(_config.DataConfigFactory):
    action_dim: int = 41

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        data_transforms = _transforms.Group(inputs=[LatentInputs(model_type=model_config.model_type)],
                                           outputs=[LatentOutputs(self.action_dim)])
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), data_transforms=data_transforms,
                                   model_transforms=_config.ModelTransformFactory()(model_config))


@dataclasses.dataclass(frozen=True)
class ShapeTolerantWeightLoader(_weight_loaders.WeightLoader):
    """openpi CheckpointWeightLoader, but parameters whose shape differs from the model (action_in_proj / action_out_proj
    when action_dim != 32) are dropped so they keep their fresh initialization."""
    params_path: str

    def load(self, params):
        loaded = _weight_loaders.CheckpointWeightLoader(self.params_path).load(params)   # merged, may contain wrong shapes
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat = flax.traverse_util.flatten_dict(loaded, sep="/")
        bad = [k for k, v in flat.items() if k in flat_ref and tuple(v.shape) != tuple(flat_ref[k].shape)]
        for k in bad:
            flat[k] = flat_ref[k]
        logging.info("ShapeTolerantWeightLoader: re-initialised %d params with a different shape: %s", len(bad), bad)
        return flax.traverse_util.unflatten_dict(flat, sep="/")


def make_accum_train_step(accum: int):
    """openpi scripts/train.py train_step with gradient accumulation inside ONE jitted step: the effective batch is split
    into `accum` micro-batches, a lax.scan runs forward+backward per micro-batch (activations of one micro-batch only) and
    adds the fp32 gradients into one carried buffer, then a single optimizer update uses their mean (= the gradient of the
    full batch). Everything after the gradient (optimizer, EMA, logged infos) is openpi's code unchanged."""
    import jax, jax.numpy as jnp, optax
    import flax.nnx as nnx
    import openpi.shared.nnx_utils as nnx_utils

    def train_step(config, rng, state, batch):
        observation, actions = batch
        split = lambda x: x.reshape((accum, x.shape[0] // accum) + x.shape[1:])
        obs_mb, act_mb = jax.tree.map(split, observation), split(actions)
        train_rng = jax.random.fold_in(rng, state.step)
        diff_state = nnx.DiffState(0, config.trainable_filter)

        def loss_fn(model, r, o, a):
            return jnp.mean(model.compute_loss(r, o, a, train=True))

        def body(carry, xs):
            acc, loss_sum = carry
            i, o, a = xs
            model = nnx.merge(state.model_def, state.params)
            model.train()
            loss, g = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, jax.random.fold_in(train_rng, i), o, a)
            return (jax.tree.map(jnp.add, acc, g), loss_sum + loss), None

        params = state.params.filter(config.trainable_filter)
        zeros = jax.tree.map(jnp.zeros_like, params)
        (acc, loss_sum), _ = jax.lax.scan(body, (zeros, jnp.zeros((), jnp.float32)), (jnp.arange(accum), obs_mb, act_mb))
        grads = jax.tree.map(lambda x: x / accum, acc)
        loss = loss_sum / accum

        # --- from here identical to openpi scripts/train.py train_step ---
        model = nnx.merge(state.model_def, state.params)
        model.train()
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)
        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(new_state, ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params))
        kernel_params = nnx.state(model, nnx.All(nnx.Param,
                                                 nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                                                 lambda _, x: x.value.ndim > 1))
        return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernel_params)}

    return train_step


def make_config(a) -> _config.TrainConfig:
    out = os.environ.get("XLVLA_PI05_OUT", "/workspace/checkpoints/dexverse/xlvla/pi05")
    return _config.TrainConfig(
        name=a.run, exp_name=a.exp,
        # max_token_len: pi0.5 writes the state as number tokens into the prompt (200 by default); longer prompts are cut with
        # only a warning, which drops the END of the state. Measured with the tokenizer (prompt sentence included): one hand
        # joint 32 -> 146, latent 41 -> 180 (fit); bimanual joint 62 -> 259, latent 82 -> 336 (do not) -> 280 / 360.
        model=pi0_config.Pi0Config(pi05=True, action_dim=a.action_dim, action_horizon=a.horizon, discrete_state_input=True,
                                   max_token_len=getattr(a, "max_token_len", None)),
        data=LatentDataConfig(repo_id=a.repo_id, base_config=_config.DataConfig(prompt_from_task=False), action_dim=a.action_dim),
        weight_loader=ShapeTolerantWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # the loader yields the effective batch (batch x accum); accum_train_step splits it into `accum` micro-batches
        batch_size=a.batch * a.accum, num_workers=a.workers,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=a.warmup, peak_lr=a.lr, decay_steps=a.steps, decay_lr=a.decay_lr),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0),
        ema_decay=a.ema if a.ema > 0 else None, num_train_steps=a.steps, seed=a.seed,
        log_interval=a.log_interval, save_interval=a.save_interval,
        # keep_period None: only the latest checkpoint stays on disk (one is ~30 GB with the train state; ten of them = 301 GB)
        keep_period=a.save_interval if getattr(a, "keep_all", False) else None,
        assets_base_dir=f"{out}/assets", checkpoint_base_dir=f"{out}/checkpoints",
        wandb_enabled=False, overwrite=a.overwrite, resume=a.resume,
        fsdp_devices=a.fsdp,   # shard params/optimizer/EMA over this many GPUs (4.2B params do not fit one 96 GB GPU with accumulation)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["norm", "train", "show"])
    ap.add_argument("--root", required=True, help="LeRobot v3.0 dataset dir (make_dataset <out>/all)")
    ap.add_argument("--repo_id", required=True, help="name of the norm-stats asset folder (assets/<run>/<repo_id>)")
    ap.add_argument("--run", required=True, help="TrainConfig name (assets/<run>, checkpoints/<run>/<exp>)")
    ap.add_argument("--exp", default="main")
    ap.add_argument("--action_dim", type=int, default=41)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_token_len", type=int, default=None,
                    help="prompt length incl. the state tokens (default 200); bimanual: 360 (latent 82) / 280 (joint 62)")
    ap.add_argument("--batch", type=int, default=32, help="micro-batch per forward/backward")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation: micro-batches per optimizer update "
                    "(effective batch = batch x accum; XL-VLA paper: 128)")
    ap.add_argument("--steps", type=int, default=20000, help="optimizer updates (each averages `accum` micro-batch gradients)")
    ap.add_argument("--decay_lr", type=float, default=1e-6)
    ap.add_argument("--fsdp", type=int, default=1, help="GPUs to shard the train state over (CUDA_VISIBLE_DEVICES picks them)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay per optimizer update; 0 = no EMA (saves one param copy, ~17 GB)")
    ap.add_argument("--lr", type=float, default=1e-4, help="peak lr (DexVerse pi0.5 setting)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=100, help="optimizer updates")
    ap.add_argument("--save_interval", type=int, default=5000, help="optimizer updates")
    ap.add_argument("--keep_all", action="store_true", help="keep every saved checkpoint (default: only the latest)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = make_config(a)
    if a.cmd == "show":
        print(cfg); return
    from dexverse_xlvla.pi05.v3_dataset import V3Dataset

    def create_torch_dataset(data_config, action_horizon, model_config):
        return V3Dataset(a.root, action_horizon)
    _data_loader.create_torch_dataset = create_torch_dataset
    openpi_root = pathlib.Path(os.environ.get("OPENPI_ROOT", "/workspace/codes/dexverse/openpi"))
    sys.path.insert(0, str(openpi_root))
    if a.cmd == "norm":   # numeric columns only (no video): state and per-frame action, as openpi's norm over chunks
        import openpi.shared.normalize as normalize
        ds = V3Dataset(a.root, cfg.model.action_horizon)
        stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
        for s in range(0, len(ds), 65536):
            stats["state"].update(ds.state[s:s + 65536]); stats["actions"].update(ds.action[s:s + 65536])
        p = cfg.assets_dirs / a.repo_id
        normalize.save(p, {k: v.get_statistics() for k, v in stats.items()})
        print("norm stats ->", p, {k: (v.get_statistics().mean[:3], v.get_statistics().q99[:3]) for k, v in stats.items()})
    else:
        import scripts.train as T
        if a.accum > 1:
            T.train_step = make_accum_train_step(a.accum)   # openpi's main jits the module-level train_step
        T.main(cfg)


if __name__ == "__main__":
    main()
