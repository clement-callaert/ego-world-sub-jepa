"""Train factored Ego-World JEPA or monolithic LeWM baseline.

Examples:
    python scripts/train.py model=factored   data=pusht
    python scripts/train.py model=monolithic data=pusht train.steps=50000
    python scripts/train.py model=factored   data=pusht train.compile=true
"""

from __future__ import annotations

import itertools
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.data import build_dataloader
from ewjepa.sigreg import latent_diagnostics
from ewjepa.train_utils import BatchPrefetcher, configure_cuda, make_adamw, maybe_compile
from ewjepa.utils import Normalizer, get_device, load_checkpoint, save_checkpoint, set_seed


def _fit_proprio_normalizer(loader, n_batches: int = 20) -> Normalizer | None:
    chunks = []
    for batch in itertools.islice(loader, n_batches):
        if "proprio" not in batch:
            return None
        chunks.append(batch["proprio"].reshape(-1, batch["proprio"].shape[-1]))
    if not chunks:
        return None
    return Normalizer.fit(torch.cat(chunks, dim=0))


def _apply_proprio_norm(proprio: torch.Tensor, normalizer: Normalizer | None) -> torch.Tensor:
    if normalizer is None:
        return proprio
    return normalizer(proprio)


def _model_state_dict(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def _training_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    normalizer: Normalizer | None,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> dict[str, torch.Tensor]:
    pixels = batch["pixels"]
    proprio = _apply_proprio_norm(batch["proprio"], normalizer)
    action = batch["action"]
    state = batch.get("state")

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model.compute_loss(pixels, proprio, action, state=state)
    scaler.scale(out["loss"]).backward()
    if grad_clip:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()
    return out


def _loss_postfix(out: dict[str, torch.Tensor], lr: float) -> dict[str, str]:
    """Short loss summary for the tqdm bar."""
    postfix = {
        "loss": f"{out['loss'].item():.3f}",
        "pred": f"{out['pred_loss'].item():.3f}",
        "sig": f"{out['sigreg'].item():.3f}",
        "lr": f"{lr:.1e}",
    }
    if out["aux_loss"].item() > 0:
        postfix["aux"] = f"{out['aux_loss'].item():.3f}"
    if out["cov_loss"].item() > 0:
        postfix["cov"] = f"{out['cov_loss'].item():.3f}"
    return postfix


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    configure_cuda(
        cudnn_benchmark=bool(cfg.train.get("cudnn_benchmark", True)),
        allow_tf32=bool(cfg.train.get("allow_tf32", True)),
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = EgoWorldConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = EgoWorldJEPA(model_cfg).to(device)
    print(f"[model] mode={model_cfg.mode} params={model.num_parameters() / 1e6:.2f}M")

    loader = build_dataloader(
        cfg.data.dataset,
        num_steps=cfg.train.num_steps,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        frameskip=cfg.train.frameskip,
        prefetch_factor=int(cfg.train.get("prefetch_factor", 4)),
        dataset_kwargs=dict(
            image_key=cfg.data.image_key,
            proprio_key=cfg.data.proprio_key,
            action_key=cfg.data.action_key,
            state_key=cfg.data.state_key,
            max_episodes=cfg.data.get("max_episodes"),
        ),
        synthetic_fallback=bool(cfg.get("synthetic_fallback", False)),
    )

    normalizer = _fit_proprio_normalizer(loader) if cfg.train.normalize_proprio else None
    opt = make_adamw(model.parameters(), cfg.train.lr, cfg.train.weight_decay, device)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    resume_path = cfg.train.get("resume_from") or (out_dir / "model.pt")
    if cfg.train.get("resume", False) and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        step = int(ckpt.get("step", 0))
        if "proprio_normalizer" in ckpt and normalizer is not None:
            normalizer = Normalizer.from_state_dict(ckpt["proprio_normalizer"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[resume] loaded {resume_path} at step {step}")

    model = maybe_compile(model, bool(cfg.train.get("compile", False)))

    run = None
    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg))

    model.train()
    prefetcher = BatchPrefetcher(loader, device)

    # Linear learning rate warmup. A deep transformer can collapse to a single
    # point if the full learning rate hits it from a cold start. We ramp the
    # rate up from 0 to base_lr over the first warmup_steps steps to avoid that.
    base_lr = float(cfg.train.lr)
    warmup_steps = max(1, int(cfg.train.get("warmup_steps", 0)))
    show_progress = bool(cfg.train.get("progress", True))

    pbar = tqdm(
        total=int(cfg.train.steps),
        initial=step,
        desc=f"train {model_cfg.mode}",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    while step < cfg.train.steps:
        lr_scale = min(1.0, (step + 1) / warmup_steps)
        for group in opt.param_groups:
            group["lr"] = base_lr * lr_scale

        batch = prefetcher.next()
        out = _training_step(
            model,
            batch,
            normalizer,
            opt,
            scaler,
            use_amp,
            float(cfg.train.grad_clip),
        )

        if not torch.isfinite(out["loss"]):
            pbar.write(f"[warn] non-finite loss at step {step}; skipping update")
            ckpt_path = out_dir / "model.pt"
            if ckpt_path.exists():
                ckpt = load_checkpoint(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model"])
                if "optimizer" in ckpt:
                    opt.load_state_dict(ckpt["optimizer"])
                pbar.write(f"[warn] reloaded checkpoint from step {ckpt.get('step', '?')}")
            step += 1
            pbar.update(1)
            continue

        current_lr = float(opt.param_groups[0]["lr"])
        pbar.set_postfix(_loss_postfix(out, current_lr), refresh=False)

        if step % cfg.train.log_every == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in out.items())
            pbar.write(f"[step {step}] {msg}")
            if run is not None:
                run.log({k: v.item() for k, v in out.items()}, step=step)

        if step % cfg.train.diag_every == 0:
            with torch.no_grad():
                proprio = _apply_proprio_norm(batch["proprio"], normalizer)
                zw, ze = model.encode_sequence(batch["pixels"], proprio)
                diag = {f"world/{k}": v for k, v in latent_diagnostics(zw.reshape(-1, zw.shape[-1])).items()}
                if ze is not None:
                    diag.update({f"ego/{k}": v for k, v in latent_diagnostics(ze.reshape(-1, ze.shape[-1])).items()})
            diag_msg = " ".join(f"{k}={v:.3f}" for k, v in diag.items())
            if model_cfg.cov_weight > 0:
                diag_msg += f" cov_loss={out['cov_loss'].item():.4f}"
            pbar.write(f"[diag {step}] {diag_msg}")
            if run is not None:
                run.log(diag, step=step)
                if model_cfg.cov_weight > 0:
                    run.log({"cov_loss": out["cov_loss"].item()}, step=step)
            world_std = diag.get("world/std", 1.0)
            world_rank = diag.get("world/effective_rank", 192.0)
            if world_std < 0.1:
                pbar.write(
                    "[warn] World latent std collapsed near zero. "
                    "Raise sigreg_mix or check the encoder."
                )
            elif world_rank < 5.0:
                pbar.write(
                    "[warn] World latent collapse detected "
                    f"(effective_rank={world_rank:.2f}). "
                    "Raise cov_weight or sigreg_mix."
                )

        if step > 0 and step % cfg.train.ckpt_every == 0:
            _save(out_dir / "model.pt", model, cfg, step, normalizer, opt, scaler if use_amp else None)

        step += 1
        pbar.update(1)

    pbar.close()
    _save(out_dir / "model.pt", model, cfg, step, normalizer, opt, scaler if use_amp else None)
    print(f"[done] checkpoint saved to {out_dir / 'model.pt'}")
    if run is not None:
        run.finish()


def _save(path, model, cfg, step, normalizer, opt=None, scaler=None) -> None:
    extra = {}
    if opt is not None:
        extra["optimizer"] = opt.state_dict()
    if scaler is not None:
        extra["scaler"] = scaler.state_dict()
    if normalizer is not None:
        extra["proprio_normalizer"] = normalizer.state_dict()
    state_dict = _model_state_dict(model)
    save_checkpoint(path, model, OmegaConf.to_container(cfg, resolve=True), step, model_state=state_dict, **extra)


if __name__ == "__main__":
    main()
