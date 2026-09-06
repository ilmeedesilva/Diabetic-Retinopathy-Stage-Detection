"""
M7 — Retinal image-quality gate.
===================================================================

A screening tool must not grade an image it cannot actually see. This module
scores an uploaded fundus photo on four cheap, interpretable signals and rejects
it (with a human-readable reason) before it ever reaches the model:

    sharpness      variance of the Laplacian   -> blurred / out of focus
    brightness     mean grey level             -> under- / over-exposed
    contrast       RMS contrast (std of grey)  -> washed out / blank
    fov_coverage   fraction of frame that is   -> not a fundus image, or the
                   retina (non-black)             retina is badly off-centre / cropped

Thresholds live in `config.yaml: quality_gate:` and are **calibrated from the
training-set distribution** in notebook 07 (default = just inside the 1st
percentile of real gradable images), so the gate rejects genuine outliers, not
normal variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Individual signals                                                          #
# --------------------------------------------------------------------------- #
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — low = blurred."""
    return float(cv2.Laplacian(_gray(img), cv2.CV_64F).var())


def brightness(img: np.ndarray) -> float:
    """Mean grey level, 0-255."""
    return float(_gray(img).mean())


def rms_contrast(img: np.ndarray) -> float:
    """Std of grey level in [0, 1] — low = washed out."""
    return float(_gray(img).astype(np.float32).std() / 255.0)


def fov_coverage(img: np.ndarray, dark_tol: int = 12) -> float:
    """Fraction of the frame brighter than `dark_tol` — an estimate of how much
    of the image is actually retina rather than black border. A true fundus photo
    is typically 0.35-0.85; a selfie / document / heavily-cropped eye is far off."""
    return float((_gray(img) > dark_tol).mean())


METRICS = {
    "sharpness": sharpness,
    "brightness": brightness,
    "contrast": rms_contrast,
    "fov_coverage": fov_coverage,
}


# --------------------------------------------------------------------------- #
# The gate                                                                    #
# --------------------------------------------------------------------------- #
# Heuristic defaults; notebook 07 overwrites these from the data distribution.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "sharpness": (8.0, float("inf")),        # reject if below
    "brightness": (25.0, 235.0),             # reject if outside
    "contrast": (0.025, float("inf")),
    "fov_coverage": (0.15, 0.98),
}

_HUMAN = {
    "sharpness": "the image is too blurred / out of focus",
    "brightness_low": "the image is too dark",
    "brightness_high": "the image is over-exposed / washed out",
    "contrast": "the image has too little contrast (looks blank or hazy)",
    "fov_coverage_low": "little of the frame looks like retina — is this a fundus photo, and is the disc/macula in view?",
    "fov_coverage_high": "the frame is almost entirely bright — likely over-exposed or not a fundus photo",
}


@dataclass
class QualityReport:
    passed: bool
    metrics: dict[str, float]
    reasons: list[str]

    def as_text(self) -> str:
        if self.passed:
            return "Image quality OK — proceeding to grading."
        return "Image rejected before grading:\n" + "\n".join(f"  - {r}" for r in self.reasons)


class QualityGate:
    def __init__(self, thresholds: dict[str, tuple[float, float]] | None = None):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "QualityGate":
        qc = cfg.get("quality_gate", {}) or {}
        th = {k: tuple(v) for k, v in qc.get("thresholds", {}).items()}
        return cls(th)

    def assess(self, img_rgb: np.ndarray) -> QualityReport:
        m = {name: fn(img_rgb) for name, fn in METRICS.items()}
        reasons: list[str] = []

        lo, hi = self.th["sharpness"]
        if m["sharpness"] < lo:
            reasons.append(_HUMAN["sharpness"])

        lo, hi = self.th["brightness"]
        if m["brightness"] < lo:
            reasons.append(_HUMAN["brightness_low"])
        elif m["brightness"] > hi:
            reasons.append(_HUMAN["brightness_high"])

        lo, hi = self.th["contrast"]
        if m["contrast"] < lo:
            reasons.append(_HUMAN["contrast"])

        lo, hi = self.th["fov_coverage"]
        if m["fov_coverage"] < lo:
            reasons.append(_HUMAN["fov_coverage_low"])
        elif m["fov_coverage"] > hi:
            reasons.append(_HUMAN["fov_coverage_high"])

        return QualityReport(passed=len(reasons) == 0, metrics=m, reasons=reasons)


# --------------------------------------------------------------------------- #
# Threshold calibration helper (used by notebook 07)                          #
# --------------------------------------------------------------------------- #
def calibrate_thresholds(images_rgb, low_pct: float = 1.0, high_pct: float = 99.0
                         ) -> dict[str, list[float]]:
    """Given an iterable of *known-gradable* RGB images, return
    {metric: [low, high]} at the given percentiles. `sharpness` and `contrast`
    get only a lower bound; `brightness` and `fov_coverage` get both."""
    vals = {k: [] for k in METRICS}
    for im in images_rgb:
        for k, fn in METRICS.items():
            vals[k].append(fn(im))
    out = {}
    for k, v in vals.items():
        v = np.asarray(v)
        lo = float(np.percentile(v, low_pct))
        hi = float(np.percentile(v, high_pct))
        if k in ("sharpness", "contrast"):
            out[k] = [round(lo, 4), float("inf")]
        else:
            out[k] = [round(lo, 4), round(hi, 4)]
    return out


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.quality_gate`                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config, load_splits

    cfg = load_config()
    df = load_splits(cfg)["test"].head(6)
    gate = QualityGate.from_config(cfg)
    for p in df["path"]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        r = gate.assess(img)
        print(f"{'PASS' if r.passed else 'FAIL'}  "
              f"sharp={r.metrics['sharpness']:.0f} bright={r.metrics['brightness']:.0f} "
              f"contrast={r.metrics['contrast']:.3f} fov={r.metrics['fov_coverage']:.2f}"
              + ("" if r.passed else f"  <- {r.reasons}"))

    # a deliberately bad image
    blank = np.full((256, 256, 3), 5, np.uint8)
    print("\nblank image ->", gate.assess(blank).as_text().replace("\n", " "))
