"""
M6 — Model evaluation & performance analysis.
===================================================================

Everything the rubric asks for, as small composable functions:

    load_fold_models      load fold{0..k}_best.pt into an ensemble
    predict_split         ensemble (+ optional TTA) predictions over a whole split
    per_class_report      precision / recall / F1 / support per stage + macro/weighted
    qwk                   quadratic weighted kappa (the clinical grading-agreement metric)
    confusion             raw + row-normalised confusion matrices
    ovr_roc_pr            one-vs-rest ROC & PR curves + AUC / AP per stage
    referable_eval        sensitivity / specificity / AUC for the refer-or-not decision
    expected_calibration_error   ECE + reliability-diagram bins
    fit_temperature       temperature scaling (optimise NLL on val logits)
    mc_dropout_predict    predictive mean + std via Monte-Carlo dropout (uncertainty)
    gradcam_plus_plus     Grad-CAM++ heat-maps for the stage head
    denormalize           tensor -> uint8 RGB for overlays

The `ordinal` head prediction (rank-consistent) is treated as the *primary* stage
estimate for order-aware metrics; the softmax head is kept for per-class metrics,
ROC/PR and Grad-CAM.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.augment import build_eval_aug, build_tta_augs
from src.data import APTOSDataset, DatasetConfig
from src.model import DRModel
from src.preprocess import make_preprocessor


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_fold_models(cfg: dict[str, Any], device: str,
                     ckpt_dir: str | Path | None = None) -> list[DRModel]:
    """Load every fold{N}_best.pt checkpoint into an ensemble of eval-mode models."""
    ckpt_dir = Path(ckpt_dir or (Path(cfg["paths"]["outputs"]) / "checkpoints"))
    paths = sorted(glob.glob(str(ckpt_dir / "fold*_best.pt")))
    if not paths:
        raise FileNotFoundError(
            f"No fold*_best.pt in {ckpt_dir}. Run notebook 05 (or download the "
            f"checkpoints from your Kaggle run) first.")
    models = []
    for p in paths:
        m = DRModel(cfg, pretrained=False).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} fold checkpoint(s): {[Path(p).name for p in paths]}")
    return models


# --------------------------------------------------------------------------- #
# Prediction                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_split(models: list[DRModel], df: pd.DataFrame, cfg: dict[str, Any],
                  device: str, tta: bool = True, batch_size: int = 32) -> dict[str, np.ndarray]:
    """Ensemble (mean over folds) + optional 4-view TTA predictions for a split.
    Returns numpy arrays keyed: y_true, stage_logits, stage_probs, stage_pred,
    ord_probs, ord_pred, ref_prob."""
    pre = make_preprocessor(cfg)
    size = cfg["image"]["size_stage2"]
    mean, std = cfg["image"]["mean"], cfg["image"]["std"]
    transforms = build_tta_augs(size, mean, std) if tta else [build_eval_aug(size, mean, std)]

    y_true = df["stage"].to_numpy()
    n, K = len(df), len(cfg["classes"])
    logit_sum = np.zeros((n, K), dtype=np.float64)
    ord_sum = np.zeros((n, K - 1), dtype=np.float64)
    ref_sum = np.zeros(n, dtype=np.float64)
    n_views = len(transforms) * len(models)

    for tf in transforms:
        ds = APTOSDataset(df.reset_index(drop=True),
                          DatasetConfig(image_size=size, preprocess=pre, transform=tf))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                             num_workers=2 if device == "cuda" else 0)
        row = 0
        for xb, _ in loader:
            xb = xb.to(device)
            bs = xb.size(0)
            for m in models:
                out = m(xb)
                logit_sum[row:row + bs] += out["stage"].float().cpu().numpy()
                ord_sum[row:row + bs] += out["ordinal"].sigmoid().float().cpu().numpy()
                ref_sum[row:row + bs] += out["referable"].sigmoid().float().cpu().numpy()
            row += bs

    stage_logits = logit_sum / n_views
    stage_probs = F.softmax(torch.tensor(stage_logits), dim=1).numpy()
    ord_probs = ord_sum / n_views
    return {
        "y_true": y_true,
        "stage_logits": stage_logits,
        "stage_probs": stage_probs,
        "stage_pred": stage_probs.argmax(1),
        "ord_probs": ord_probs,
        "ord_pred": (ord_probs > 0.5).sum(1),
        "ref_prob": ref_sum / n_views,
    }


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def qwk(y_true, y_pred, num_classes: int = 5) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                   labels=list(range(num_classes))))


def per_class_report(y_true, y_pred, classes: dict[int, str]) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), zero_division=0)
    rows = [{"stage": k, "name": classes[k], "precision": p[i], "recall": r[i],
             "f1": f[i], "support": int(s[i])} for i, k in enumerate(classes)]
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(classes), average=avg, zero_division=0)
        rows.append({"stage": "", "name": f"{avg} avg", "precision": pa,
                     "recall": ra, "f1": fa, "support": len(y_true)})
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, num_classes: int = 5):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm, norm


def ovr_roc_pr(y_true, probs, num_classes: int = 5) -> dict[int, dict]:
    """One-vs-rest ROC & PR per stage."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    out = {}
    for k in range(num_classes):
        yk = (np.asarray(y_true) == k).astype(int)
        if yk.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yk, probs[:, k])
        prec, rec, _ = precision_recall_curve(yk, probs[:, k])
        out[k] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(yk, probs[:, k]),
                  "prec": prec, "rec": rec, "ap": average_precision_score(yk, probs[:, k])}
    return out


def referable_eval(y_true_stage, ref_prob, refer_threshold: int = 2,
                   operating_point: float = 0.5) -> dict:
    """Binary 'refer to ophthalmologist' (stage >= threshold) analysis."""
    from sklearn.metrics import roc_curve, roc_auc_score
    y = (np.asarray(y_true_stage) >= refer_threshold).astype(int)
    fpr, tpr, thr = roc_curve(y, ref_prob)
    auc = roc_auc_score(y, ref_prob)
    pred = (np.asarray(ref_prob) >= operating_point).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"auc": float(auc), "sensitivity": sens, "specificity": spec,
            "fpr": fpr, "tpr": tpr, "thr": thr, "operating_point": operating_point,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


# --------------------------------------------------------------------------- #
# Calibration                                                                 #
# --------------------------------------------------------------------------- #
def expected_calibration_error(probs, y_true, n_bins: int = 15):
    """ECE + per-bin (confidence, accuracy, count) for a reliability diagram."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append(((lo + hi) / 2, np.nan, 0))
            continue
        acc_b, conf_b, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc_b - conf_b)
        rows.append(((lo + hi) / 2, acc_b, int(m.sum())))
    return float(ece), rows


def fit_temperature(logits, y_true, max_iter: int = 100) -> float:
    """Optimise a single scalar T minimising NLL on held-out logits (Guo et al. 2017)."""
    logits = torch.tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_true), dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3))


# --------------------------------------------------------------------------- #
# Uncertainty — Monte-Carlo dropout                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mc_dropout_predict(model: DRModel, x: torch.Tensor, passes: int = 20):
    """Keep Dropout active (BatchNorm frozen) and run `passes` forward passes.
    Returns (mean_probs [B,K], predictive_std [B], predictive_entropy [B])."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    probs = torch.stack([model(x)["stage"].softmax(1) for _ in range(passes)])  # (P,B,K)
    mean = probs.mean(0)
    std = probs.std(0).gather(1, mean.argmax(1, keepdim=True)).squeeze(1)       # std of top class
    ent = -(mean.clamp_min(1e-8) * mean.clamp_min(1e-8).log()).sum(1)
    return mean.cpu().numpy(), std.cpu().numpy(), ent.cpu().numpy()


# --------------------------------------------------------------------------- #
# Grad-CAM++                                                                  #
# --------------------------------------------------------------------------- #
class _StageLogits(nn.Module):
    """Adapter: pytorch-grad-cam expects model(x) -> class-logit tensor."""

    def __init__(self, model: DRModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["stage"]


def gradcam_plus_plus(model: DRModel, images: torch.Tensor, target_classes,
                      device: str) -> np.ndarray:
    """Return (B, H, W) Grad-CAM++ maps in [0,1] for the stage head, at the
    CBAM output layer."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAMPlusPlus(model=_StageLogits(model).to(device),
                          target_layers=[model.cam_target_layer])
    targets = [ClassifierOutputTarget(int(c)) for c in target_classes]
    return cam(input_tensor=images.to(device), targets=targets)


def denormalize(t: torch.Tensor, mean, std) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) uint8 RGB."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    x = (t.detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.evaluate`  (random-init model, tiny split)       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config, load_splits

    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    test = load_splits(cfg)["test"].sample(24, random_state=0).reset_index(drop=True)
    model = DRModel(cfg, pretrained=False).to(dev).eval()

    pred = predict_split([model], test, cfg, dev, tta=False, batch_size=8)
    print("stage QWK :", qwk(pred["y_true"], pred["ord_pred"]))
    print(per_class_report(pred["y_true"], pred["stage_pred"], cfg["classes"]).round(3).to_string(index=False))
    cm, cmn = confusion(pred["y_true"], pred["stage_pred"])
    print("confusion:\n", cm)
    print("referable:", {k: v for k, v in referable_eval(pred["y_true"], pred["ref_prob"]).items()
                         if k in ("auc", "sensitivity", "specificity")})
    ece, _ = expected_calibration_error(pred["stage_probs"], pred["y_true"])
    print("ECE:", round(ece, 4), "| T*:", round(fit_temperature(pred["stage_logits"], pred["y_true"]), 3))
