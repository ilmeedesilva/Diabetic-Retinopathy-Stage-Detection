"""
M2 — Fundus image preprocessing pipeline.
===================================================================

A retinal fundus photo is mostly uninformative black border, has strong and
uneven illumination from the camera flash, and low local contrast where the
clinically important lesions are (microaneurysms, dot/blot haemorrhages, hard
exudates). This module turns a raw photo into a clean, illumination-normalised,
contrast-enhanced image that a CNN can learn from more easily.

Pipeline (each step is a standalone function so it can be toggled / ablated):

    raw RGB
      -> crop_dark_borders   remove the black frame around the retina
      -> resize(work_size)   fixed working resolution so later kernels are scale-consistent
      -> circle_mask         keep only the circular field of view
      -> ben_graham          subtract a local-average blur -> removes lighting/colour cast
      -> clahe_lab           CLAHE on L* of LAB -> local contrast, colour preserved
      -> circle_mask (again) Ben Graham fills the corners with grey; re-mask to black
    -> preprocessed RGB (uint8, work_size x work_size)

`clahe_green()` is provided separately as an *analysis view* — the green channel
carries the highest lesion/vessel contrast in fundus imaging and is shown in the
report, and could be added as an extra input channel later.

No parameters are fitted on the data, and every operation is per-image and
deterministic, so this stage introduces **no train/test leakage**.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Individual steps                                                            #
# --------------------------------------------------------------------------- #
def crop_dark_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the near-black frame: bounding box of pixels brighter than `tol`."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() < 100:                       # essentially blank image -> leave as is
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    # guard against a degenerate sliver
    return crop if min(crop.shape[:2]) >= 32 else img


def circle_mask(img: np.ndarray, radius_scale: float = 0.97) -> np.ndarray:
    """Zero everything outside the largest centred circle (the fundus FOV).

    `radius_scale` < 1 insets the circle slightly to drop the bright rim / lens
    vignette that otherwise looks like a giant edge to the network.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * radius_scale)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, thickness=-1)
    out = img.copy()
    out[m == 0] = 0
    return out


def ben_graham(img: np.ndarray, sigma_frac: float = 0.1, alpha: float = 4.0,
               beta: float = -4.0, gamma: float = 128.0) -> np.ndarray:
    """Ben Graham's normalisation (winner, 2015 Kaggle DR competition).

        out = alpha * img + beta * GaussianBlur(img) + gamma

    Subtracting a heavily blurred copy removes the smooth, low-frequency
    illumination and colour cast from the flash, leaving the local structure
    (vessels, lesions) on a flat mid-grey background.
    """
    h, w = img.shape[:2]
    sigma = max(h, w) * sigma_frac
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img.astype(np.float32), alpha, blur.astype(np.float32), beta, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the L* channel of
    LAB, so contrast is boosted locally without shifting hue/saturation."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def clahe_green(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Analysis view: CLAHE on the green channel (highest lesion contrast).
    Returns a single-channel HxW uint8 image."""
    g = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(g)


# --------------------------------------------------------------------------- #
# Full pipeline                                                               #
# --------------------------------------------------------------------------- #
def preprocess_image(
    img: np.ndarray,
    work_size: int = 512,
    do_crop: bool = True,
    do_circle: bool = True,
    circle_radius_scale: float = 0.97,
    do_ben: bool = True,
    ben_params: dict | None = None,
    do_clahe: bool = True,
    clahe_params: dict | None = None,
    return_steps: bool = False,
):
    """Run the ordered pipeline on one RGB uint8 image.

    If `return_steps` is True, also return a dict of the intermediate images
    (used by notebook 02 to draw the step-by-step figure).
    """
    steps: dict[str, np.ndarray] = {"original": img.copy()}

    if do_crop:
        img = crop_dark_borders(img)
        steps["cropped"] = img.copy()

    img = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_AREA)

    if do_circle:
        img = circle_mask(img, circle_radius_scale)
        steps["circle_masked"] = img.copy()

    if do_ben:
        img = ben_graham(img, **(ben_params or {}))
        steps["ben_graham"] = img.copy()

    if do_clahe:
        img = clahe_lab(img, **(clahe_params or {}))
        steps["clahe"] = img.copy()

    if do_circle:
        # Ben Graham's gamma term turns the masked corners grey — re-mask them.
        img = circle_mask(img, circle_radius_scale)

    steps["final"] = img.copy()
    return (img, steps) if return_steps else img


class Preprocessor:
    """Picklable `callable(rgb_uint8) -> rgb_uint8` built from the `preprocess:`
    block of config.yaml. A class (not a closure) so it survives being sent to
    DataLoader worker processes in M5."""

    def __init__(self, cfg: dict[str, Any]):
        pc = (cfg.get("preprocess", {}) or {})
        self.work_size = pc.get("work_size", 512)
        self.do_crop = pc.get("crop", True)
        self.do_circle = pc.get("circle_mask", True)
        self.circle_radius_scale = pc.get("circle_radius_scale", 0.97)
        self.do_ben = pc.get("ben_graham", True)
        self.ben_params = pc.get("ben_params", {}) or {}
        self.do_clahe = pc.get("clahe", True)
        self.clahe_params = pc.get("clahe_params", {}) or {}

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return preprocess_image(
            img, work_size=self.work_size, do_crop=self.do_crop,
            do_circle=self.do_circle, circle_radius_scale=self.circle_radius_scale,
            do_ben=self.do_ben, ben_params=self.ben_params,
            do_clahe=self.do_clahe, clahe_params=self.clahe_params,
        )


def make_preprocessor(cfg: dict[str, Any]) -> "Preprocessor":
    """Backward-compatible factory used by every notebook."""
    return Preprocessor(cfg)


# --------------------------------------------------------------------------- #
# Image-quality metrics (evidence that preprocessing helps; reused by M7 gate) #
# --------------------------------------------------------------------------- #
def rms_contrast(img: np.ndarray) -> float:
    """RMS contrast = std of grayscale intensities in [0, 1]. Higher = more contrast."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float(g.std())


def shannon_entropy(img: np.ndarray) -> float:
    """Grayscale histogram entropy (bits). Higher = better tonal spread."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — the focus/blur measure the M7 gate thresholds."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.preprocess`                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config, load_labels

    cfg = load_config()
    df = load_labels(cfg)
    path = df["path"].iloc[0]
    raw = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    pre = make_preprocessor(cfg)(raw)
    print(f"{path}")
    print(f"raw   {raw.shape}  contrast={rms_contrast(raw):.4f}  entropy={shannon_entropy(raw):.3f}")
    print(f"proc  {pre.shape}  contrast={rms_contrast(pre):.4f}  entropy={shannon_entropy(pre):.3f}")
