"""
M5 — Training strategy: two-phase fine-tuning + 3-fold stratified CV.
===================================================================

Strategy (each choice is deliberate — see notebook 05 for the justification):

* **Phase 1 — frozen backbone.** Only CBAM + the three heads train, at the small
  progressive-resizing size (`image.size_stage1`). Fast, and prevents the randomly
  initialised heads from wrecking the pretrained backbone with large early gradients.
* **Phase 2 — full fine-tune.** Backbone unfrozen with a **discriminative learning
  rate** (`lr_backbone` << `lr_head`), at the larger size (`image.size_stage2`,
  progressive resizing), with **cosine-annealing warm restarts** and **early
  stopping** on validation QWK.
* **Mixed precision** on CUDA (a no-op on MPS/CPU, which run fp32).
* **MixUp / CutMix** applied per-batch (train split only) via `src.augment`.
* **3-fold stratified CV** (`src.data.stratified_folds`) so results are reported
  as mean ± std, not a single lucky split.
* **Checkpointing on validation QWK**, not accuracy — QWK is the order-aware,
  clinically meaningful metric (see `src.losses.qwk_loss`).
* **Test-time augmentation** (`predict_tta`) averages 4 deterministic views
  (identity / H-flip / V-flip / 180°) at inference.

Every knob can be overridden per-call (see `cross_validate`) without touching
`config.yaml`, so the exact same code runs a 2-minute smoke test on a laptop CPU
and the full report-quality run on a GPU.
"""

from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader

from src.augment import (build_eval_aug, build_train_aug, build_tta_augs,
                         cutmix_batch, make_weighted_sampler, mixup_batch,
                         oversample_minorities, per_class_weights)
from src.data import APTOSDataset, DatasetConfig, stratified_folds
from src.losses import MultiTaskLoss
from src.model import DRModel, predict_stage
from src.preprocess import make_preprocessor


# --------------------------------------------------------------------------- #
# Device / AMP helpers                                                        #
# --------------------------------------------------------------------------- #
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _autocast(device: str, enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled) if device == "cuda" else nullcontext()


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_qwk(preds: np.ndarray, targets: np.ndarray, num_classes: int = 5) -> float:
    return float(cohen_kappa_score(targets, preds, weights="quadratic",
                                   labels=list(range(num_classes))))


def _safe_save(state: dict, path: Path, log: Callable[[str], None]) -> None:
    """torch.save that degrades to a warning instead of killing a multi-hour run
    (e.g. on a transient full-disk error). The best weights stay in memory and
    are loaded into the returned model regardless of whether the file lands."""
    try:
        torch.save(state, path)
    except (OSError, RuntimeError) as e:
        # torch's zip writer wraps a full-disk error as RuntimeError, not OSError
        log(f"  WARNING: could not write checkpoint to {path} ({e}). "
            f"Training continues; free disk space and re-run to persist it.")


# --------------------------------------------------------------------------- #
# Optimiser — discriminative learning rates                                   #
# --------------------------------------------------------------------------- #
def build_optimizer(model: DRModel, lr_backbone: float, lr_head: float,
                    weight_decay: float, freeze_backbone: bool) -> torch.optim.Optimizer:
    backbone_params, head_params = model.group_parameters()
    for p in backbone_params:
        p.requires_grad = not freeze_backbone
    groups = [{"params": [p for p in head_params if p.requires_grad], "lr": lr_head}]
    if not freeze_backbone:
        groups.append({"params": [p for p in backbone_params if p.requires_grad], "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# --------------------------------------------------------------------------- #
# One epoch                                                                   #
# --------------------------------------------------------------------------- #
def run_epoch(model: DRModel, loader: DataLoader, criterion: MultiTaskLoss, device: str,
             optimizer: torch.optim.Optimizer | None = None,
             scaler: torch.cuda.amp.GradScaler | None = None,
             mix_cfg: dict[str, float] | None = None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = bool(scaler is not None and scaler.is_enabled())
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = {k: v.to(device, non_blocking=True) for k, v in yb.items()}

            targets_b, lam = None, 1.0
            if is_train and mix_cfg and np.random.rand() < mix_cfg.get("mix_prob", 0.0):
                if np.random.rand() < 0.5:
                    xb, idx, lam = mixup_batch(xb, mix_cfg.get("mixup_alpha", 0.2))
                else:
                    xb, idx, lam = cutmix_batch(xb, mix_cfg.get("cutmix_alpha", 1.0))
                targets_b = {k: v[idx] for k, v in yb.items()}

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _autocast(device, use_amp):
                out = model(xb)
                loss, _ = criterion(out, yb, targets_b=targets_b, lam=lam)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = xb.size(0)
            total_loss += float(loss.detach()) * bs
            n += bs
            pred = predict_stage(out)
            all_preds.append(pred["ordinal_pred"].detach().cpu().numpy())
            all_targets.append(yb["stage"].detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / max(n, 1),
        "acc": float(accuracy_score(targets, preds)),
        "qwk": compute_qwk(preds, targets),
    }


# --------------------------------------------------------------------------- #
# One fold: phase 1 (frozen) -> phase 2 (fine-tune, discriminative LR, cosine) #
# --------------------------------------------------------------------------- #
def train_one_fold(train_df, val_df, cfg: dict[str, Any], device: str, fold: int = 0,
                   log: Callable[[str], None] = print) -> tuple[DRModel, list[dict], float]:
    tcfg, acfg, icfg = cfg["train"], cfg["augment"], cfg["image"]
    mean, std = icfg["mean"], icfg["std"]
    pre = make_preprocessor(cfg)
    num_classes = len(cfg["classes"])

    # M3 balancing strategy applied to THIS fold's training rows only.
    phase_train_df = train_df
    if acfg["balancing"] == "oversample":
        phase_train_df = oversample_minorities(
            train_df, target=acfg["oversample_target"],
            cap_ratio=acfg["oversample_cap_ratio"], seed=cfg["seed"])

    cw = torch.tensor(per_class_weights(train_df, num_classes, beta=acfg["class_weight_beta"]),
                      dtype=torch.float32, device=device)
    model = DRModel(cfg).to(device)
    criterion = MultiTaskLoss(cfg, class_weight=cw)
    mix_cfg = {"mix_prob": tcfg.get("mix_prob", 0.5),
              "mixup_alpha": tcfg["mixup_alpha"], "cutmix_alpha": tcfg["cutmix_alpha"]}

    def make_loaders(size: int):
        train_tf = build_train_aug(size, mean, std, strength=acfg["strength"])
        eval_tf = build_eval_aug(size, mean, std)
        train_ds = APTOSDataset(phase_train_df.reset_index(drop=True),
                                DatasetConfig(image_size=size, preprocess=pre, transform=train_tf))
        val_ds = APTOSDataset(val_df.reset_index(drop=True),
                              DatasetConfig(image_size=size, preprocess=pre, transform=eval_tf))
        nw = 2 if device == "cuda" else 0        # avoid notebook multiprocessing issues off-GPU
        sampler, shuffle = None, True
        if acfg["balancing"] == "weighted_sampler":
            sampler, shuffle = make_weighted_sampler(phase_train_df, num_classes), False
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=shuffle,
                                  sampler=sampler, num_workers=nw, pin_memory=(device == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=nw)
        return train_loader, val_loader

    ckpt_dir = Path(cfg["paths"]["outputs"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_qwk, best_state = -1.0, None
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and tcfg["amp"]))

    # ---- Phase 1: frozen backbone, heads only -------------------------- #
    train_loader, val_loader = make_loaders(icfg["size_stage1"])
    optimizer = build_optimizer(model, tcfg["lr_backbone"], tcfg["lr_head"],
                               tcfg["weight_decay"], freeze_backbone=True)
    for epoch in range(tcfg["phase1_epochs"]):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, mix_cfg)
        va = run_epoch(model, val_loader, criterion, device)
        history.append({"phase": 1, "epoch": epoch, "train": tr, "val": va, "lr": tcfg["lr_head"]})
        log(f"[fold {fold} | phase1 {epoch+1}/{tcfg['phase1_epochs']}] "
            f"train_loss={tr['loss']:.3f} val_loss={va['loss']:.3f} "
            f"val_acc={va['acc']:.3f} val_qwk={va['qwk']:.3f}  ({time.time()-t0:.1f}s)")
        if va["qwk"] > best_qwk:
            best_qwk, best_state = va["qwk"], copy.deepcopy(model.state_dict())
            _safe_save(best_state, ckpt_dir / f"fold{fold}_best.pt", log)

    # ---- Phase 2: unfreeze, discriminative LR, cosine restarts --------- #
    train_loader, val_loader = make_loaders(icfg["size_stage2"])
    optimizer = build_optimizer(model, tcfg["lr_backbone"], tcfg["lr_head"],
                               tcfg["weight_decay"], freeze_backbone=False)
    epochs2 = tcfg["phase2_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs2 // 3))
    patience, no_improve = tcfg["early_stopping_patience"], 0

    for epoch in range(epochs2):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, mix_cfg)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"phase": 2, "epoch": epoch, "train": tr, "val": va,
                        "lr": optimizer.param_groups[-1]["lr"]})
        log(f"[fold {fold} | phase2 {epoch+1}/{epochs2}] "
            f"train_loss={tr['loss']:.3f} val_loss={va['loss']:.3f} "
            f"val_acc={va['acc']:.3f} val_qwk={va['qwk']:.3f}  ({time.time()-t0:.1f}s)")
        if va["qwk"] > best_qwk:
            best_qwk, best_state, no_improve = va["qwk"], copy.deepcopy(model.state_dict()), 0
            _safe_save(best_state, ckpt_dir / f"fold{fold}_best.pt", log)
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"[fold {fold}] early stopping (no val QWK improvement in {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_qwk


# --------------------------------------------------------------------------- #
# K-fold cross-validation                                                     #
# --------------------------------------------------------------------------- #
def cross_validate(train_df, cfg: dict[str, Any], device: str,
                   n_folds: int | None = None, phase1_epochs: int | None = None,
                   phase2_epochs: int | None = None, size_stage1: int | None = None,
                   size_stage2: int | None = None, subset: int | None = None,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stratified K-fold CV. Every override is a *local copy* of cfg — config.yaml
    on disk is never touched, so the same call runs a fast smoke test (small
    `subset`, `n_folds=2`, `phase*_epochs=1`, small sizes) or the full report run
    (defaults from config.yaml) without editing anything."""
    cfg = copy.deepcopy(cfg)
    if n_folds is not None:
        cfg["split"]["n_folds"] = n_folds
    if phase1_epochs is not None:
        cfg["train"]["phase1_epochs"] = phase1_epochs
    if phase2_epochs is not None:
        cfg["train"]["phase2_epochs"] = phase2_epochs
    if size_stage1 is not None:
        cfg["image"]["size_stage1"] = size_stage1
    if size_stage2 is not None:
        cfg["image"]["size_stage2"] = size_stage2

    df = train_df if subset is None else (
        train_df.sample(subset, random_state=cfg["seed"]).reset_index(drop=True))

    results = []
    for fold, tr_idx, va_idx in stratified_folds(df, cfg):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        log(f"=== Fold {fold}/{cfg['split']['n_folds']}  train={len(tr_df)}  val={len(va_df)} ===")
        _, history, best_qwk = train_one_fold(tr_df, va_df, cfg, device, fold=fold, log=log)
        results.append({"fold": fold, "best_val_qwk": best_qwk, "history": history})

    qwks = [r["best_val_qwk"] for r in results]
    summary = {
        "n_folds": len(results),
        "val_qwk_mean": float(np.mean(qwks)),
        "val_qwk_std": float(np.std(qwks)),
        "folds": results,
        "config_used": {"phase1_epochs": cfg["train"]["phase1_epochs"],
                        "phase2_epochs": cfg["train"]["phase2_epochs"],
                        "size_stage1": cfg["image"]["size_stage1"],
                        "size_stage2": cfg["image"]["size_stage2"],
                        "n_folds": cfg["split"]["n_folds"],
                        "subset": subset},
    }
    out = Path(cfg["paths"]["outputs"]) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    try:
        with open(out / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except OSError as e:
        log(f"WARNING: could not write {out / 'cv_summary.json'} ({e}); "
            f"summary is still returned in memory.")
    log(f"\nCV done: val QWK = {summary['val_qwk_mean']:.3f} +/- {summary['val_qwk_std']:.3f}")
    return summary


# --------------------------------------------------------------------------- #
# Test-time augmentation inference (used here and reused in M6)               #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_tta(model: DRModel, img_rgb: np.ndarray, cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Average predictions over 4 deterministic, label-preserving views
    (identity / H-flip / V-flip / 180 deg)."""
    model.eval()
    size = cfg["image"]["size_stage2"]
    ttas = build_tta_augs(size, cfg["image"]["mean"], cfg["image"]["std"])
    soft_probs, ord_probs = [], []
    for tf in ttas:
        x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
        pred = predict_stage(model(x))
        soft_probs.append(pred["softmax_prob"].cpu().numpy()[0])
        ord_probs.append(pred["ordinal_prob"].cpu().numpy()[0])
    avg_soft = np.mean(soft_probs, axis=0)
    avg_ord = np.mean(ord_probs, axis=0)
    return {
        "softmax_pred": int(np.argmax(avg_soft)), "softmax_prob": avg_soft,
        "ordinal_pred": int((avg_ord > 0.5).sum()), "ordinal_prob": avg_ord,
    }


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.train`  (tiny CV run, proves the loop is correct) #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config, load_splits

    cfg = load_config()
    device = get_device()
    print("device:", device)
    train_df = load_splits(cfg)["train"]

    summary = cross_validate(
        train_df, cfg, device,
        n_folds=2, phase1_epochs=1, phase2_epochs=1,
        size_stage1=96, size_stage2=96, subset=150,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))
