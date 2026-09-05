"""
M3 — Data augmentation and class-imbalance handling.
===================================================================

Two problems, addressed separately:

1. **Limited diversity.** 2.5k training images of one camera type. We expand
   apparent diversity with augmentations that are *label-preserving for fundus
   images* — see `build_train_aug` and the INCLUDED / EXCLUDED reasoning below.

2. **Severe class imbalance** (~9.4 : 1, No-DR vs Severe). Three independent
   strategies are provided so notebook 03 can compare them:
     * `per_class_weights`      — cost-sensitive loss weights (incl. the
       "effective number of samples" scheme, Cui et al. 2019)
     * `make_weighted_sampler`  — oversample minorities at batch-draw time
     * `oversample_minorities`  — physically duplicate minority rows

Plus `mixup_batch` / `cutmix_batch` regularisers (used in M5).

INCLUDED augmentations and why they keep the DR grade valid
----------------------------------------------------------
* H/V flips, 90 deg rotations   — the retina has no canonical orientation.
* small affine (<=25 deg, <=10% shift/scale) — models patient/camera pose.
* mild elastic / grid / optical distortion — models lens and curvature variation.
* +/-10% brightness & contrast   — models exposure differences between clinics.
* small CoarseDropout            — occlusion robustness (dust, eyelashes).

EXCLUDED on purpose
-------------------
* large hue/saturation shifts, channel shuffle — colour *is* signal: hard
  exudates are yellow, haemorrhages deep red. Distorting it corrupts the label.
* heavy blur / downscale — would erase microaneurysms, the earliest DR sign.
* vertical perspective / large crops — can crop out the macula or optic disc,
  which changes the correct grade.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import WeightedRandomSampler

from src.data import class_weights


# --------------------------------------------------------------------------- #
# Augmentation pipelines                                                      #
# --------------------------------------------------------------------------- #
def build_train_aug(size: int, mean, std, strength: float = 1.0,
                    normalize: bool = True) -> A.Compose:
    """Training-time augmentation. `strength` scales every stochastic p in [0, 1+].
    `normalize=False` returns uint8 HxWxC images for visualisation."""
    s = strength
    tfs = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-25, 25),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7 * s),
        A.OneOf([
            A.ElasticTransform(alpha=1.0, sigma=20, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
        ], p=0.30 * s),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5 * s),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(0.03, 0.08),
                        hole_width_range=(0.03, 0.08),
                        fill=0, p=0.30 * s),
    ]
    if normalize:
        tfs += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(tfs)


def build_eval_aug(size: int, mean, std) -> A.Compose:
    """Deterministic val/test transform: resize + normalise only."""
    return A.Compose([A.Resize(size, size), A.Normalize(mean=mean, std=std), ToTensorV2()])


def build_tta_augs(size: int, mean, std) -> list[A.Compose]:
    """Four deterministic, label-preserving views for test-time augmentation (M6):
    identity, H-flip, V-flip, 180 deg."""
    tail = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return [
        A.Compose([A.Resize(size, size), *tail]),
        A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0), *tail]),
        A.Compose([A.Resize(size, size), A.VerticalFlip(p=1.0), *tail]),
        A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), *tail]),
    ]


# --------------------------------------------------------------------------- #
# Class-imbalance strategy 1 — cost-sensitive loss weights                    #
# --------------------------------------------------------------------------- #
def per_class_weights(train_df: pd.DataFrame, num_classes: int = 5,
                      beta: float | None = 0.9999) -> np.ndarray:
    """Per-class loss weights, mean-normalised to 1.0.

    beta is None -> plain inverse frequency (see src.data.class_weights).
    beta in (0,1) -> "effective number of samples": w_c ∝ (1-beta)/(1-beta^n_c)
                     (Cui et al., CVPR 2019). Handles heavy imbalance more
                     gently than raw inverse frequency.
    """
    if beta is None:
        return class_weights(train_df, num_classes)
    counts = (train_df["stage"].value_counts()
              .reindex(range(num_classes), fill_value=0).to_numpy().astype(float))
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.clip(eff_num, 1e-12, None)
    return (w / w.mean()).astype("float32")


# --------------------------------------------------------------------------- #
# Class-imbalance strategy 2 — weighted sampler                               #
# --------------------------------------------------------------------------- #
def make_weighted_sampler(train_df: pd.DataFrame, num_classes: int = 5) -> WeightedRandomSampler:
    """Draw minority-class rows more often so each batch is ~class-balanced.
    Use ONLY on the training set."""
    w = class_weights(train_df, num_classes)                 # per class
    sample_w = np.array([float(w[int(c)]) for c in train_df["stage"]], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w), replacement=True,
    )


# --------------------------------------------------------------------------- #
# Class-imbalance strategy 3 — physical oversampling of minority rows         #
# --------------------------------------------------------------------------- #
def oversample_minorities(train_df: pd.DataFrame, target: str | int = "median",
                          cap_ratio: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Duplicate minority-class rows (with replacement) up to `target` count.

    target: "median" | "max" | int. cap_ratio limits target to
    cap_ratio * smallest_class so we don't 10x-duplicate tiny classes.
    Returns a shuffled DataFrame. Use ONLY on the training set.
    """
    counts = train_df["stage"].value_counts()
    tgt = int(counts.median()) if target == "median" else \
          int(counts.max()) if target == "max" else int(target)
    if cap_ratio is not None:
        tgt = min(tgt, int(counts.min() * cap_ratio))

    parts = []
    for _, grp in train_df.groupby("stage"):
        if len(grp) < tgt:
            extra = grp.sample(tgt - len(grp), replace=True, random_state=seed)
            parts.append(pd.concat([grp, extra], ignore_index=True))
        else:
            parts.append(grp)
    return (pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=seed).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# MixUp / CutMix (batch-level regularisers, used in M5)                       #
# --------------------------------------------------------------------------- #
def _rngs(seed: int | None):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return np.random.default_rng(seed), g


def mixup_batch(x: torch.Tensor, alpha: float = 0.2, seed: int | None = None,
                lam: float | None = None):
    """Convex blend of the batch with a shuffled copy.
    Returns (mixed_x, shuffle_index, lam) for a mix-aware loss:
        loss = lam * L(pred, y) + (1 - lam) * L(pred, y[index])
    Pass `lam` to force a specific blend (used for figures / tests).
    """
    npr, g = _rngs(seed)
    if lam is None:
        lam = float(npr.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), generator=g)
    return lam * x + (1.0 - lam) * x[idx], idx, lam


def _rand_bbox(h: int, w: int, lam: float, npr) -> tuple[int, int, int, int]:
    cut = np.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = int(npr.integers(w)), int(npr.integers(h))
    x0, x1 = np.clip([cx - cw // 2, cx + cw // 2], 0, w)
    y0, y1 = np.clip([cy - ch // 2, cy + ch // 2], 0, h)
    return int(x0), int(y0), int(x1), int(y1)


def cutmix_batch(x: torch.Tensor, alpha: float = 1.0, seed: int | None = None,
                 lam: float | None = None):
    """Paste a rectangular patch from a shuffled copy of the batch.
    Returns (cutmixed_x, shuffle_index, lam) where lam is the *area* kept from x.
    Pass `lam` to force the patch size (used for figures / tests).
    """
    npr, g = _rngs(seed)
    if lam is None:
        lam = float(npr.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), generator=g)
    _, _, h, w = x.shape
    x0, y0, x1, y1 = _rand_bbox(h, w, lam, npr)
    out = x.clone()
    out[:, :, y0:y1, x0:x1] = x[idx, :, y0:y1, x0:x1]
    lam = 1.0 - ((x1 - x0) * (y1 - y0) / (h * w))
    return out, idx, lam


# --------------------------------------------------------------------------- #
def build_from_config(cfg: dict[str, Any], size: int, split: str = "train"):
    """Convenience: return the right Compose for a split, driven by config."""
    mean, std = cfg["image"]["mean"], cfg["image"]["std"]
    if split == "train":
        return build_train_aug(size, mean, std,
                               strength=cfg.get("augment", {}).get("strength", 1.0))
    return build_eval_aug(size, mean, std)


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.augment`                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config, load_splits

    cfg = load_config()
    tr = load_splits(cfg)["train"]
    print("train class counts:\n", tr["stage"].value_counts().sort_index().to_string())

    print("\ninv-freq weights      :", class_weights(tr, 5).round(3))
    print("effective-num weights :", per_class_weights(tr, 5, beta=0.9999).round(3))

    os_df = oversample_minorities(tr, target="median", seed=cfg["seed"])
    print("\nafter oversample(median):\n", os_df["stage"].value_counts().sort_index().to_string())

    x = torch.rand(6, 3, 64, 64)
    mx, i, lam = mixup_batch(x, 0.2, seed=0);  print(f"\nmixup  lam={lam:.3f}  out={tuple(mx.shape)}")
    cx, i, lam = cutmix_batch(x, 1.0, seed=0); print(f"cutmix lam={lam:.3f}  out={tuple(cx.shape)}")

    aug = build_train_aug(384, cfg["image"]["mean"], cfg["image"]["std"])
    dummy = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    print("train aug tensor out:", tuple(aug(image=dummy)["image"].shape))
