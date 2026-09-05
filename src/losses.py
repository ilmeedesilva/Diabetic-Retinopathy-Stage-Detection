"""
M4 — Loss functions for the multi-task, order-aware model.
===================================================================

Four ingredients, combined by `MultiTaskLoss` with weights from
`config.yaml: train.loss_weights`:

* `FocalLoss`  on the 5-way stage head  — down-weights easy majority-class
  examples so the rare Severe / Proliferative grades still drive the gradient.
* `coral_loss` on the ordinal head      — binary cross-entropy over the K-1
  cumulative "is stage > k ?" targets (CORAL, Cao et al. 2020). Rank-consistent.
* BCE         on the referable head     — the clinical refer / no-refer decision.
* `qwk_loss`  on the stage head         — a differentiable surrogate for
  **Quadratic Weighted Kappa**, the metric ophthalmologists use to measure
  grading agreement. Optimising it directly (not just accuracy) is the
  "above and beyond" objective for this project.

`MultiTaskLoss` also supports MixUp / CutMix: pass `targets_b` and `lam` and the
loss becomes  lam * L(targets) + (1 - lam) * L(targets_b).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Focal loss (multi-class)                                                    #
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# --------------------------------------------------------------------------- #
# CORAL ordinal loss                                                         #
# --------------------------------------------------------------------------- #
def stage_to_levels(stage: torch.Tensor, num_classes: int) -> torch.Tensor:
    """stage k -> cumulative binary targets [1{k>0}, 1{k>1}, ...]  shape (B, K-1)."""
    ks = torch.arange(num_classes - 1, device=stage.device).view(1, -1)
    return (stage.view(-1, 1) > ks).float()


def coral_loss(logits: torch.Tensor, levels: torch.Tensor,
               importance: torch.Tensor | None = None) -> torch.Tensor:
    """BCE over the K-1 ordinal cut-points.  logits, levels: (B, K-1)."""
    log_p = F.logsigmoid(logits)
    log_1m_p = F.logsigmoid(logits) - logits          # = log(1 - sigmoid(logits))
    per_cut = -(levels * log_p + (1.0 - levels) * log_1m_p)
    if importance is not None:
        per_cut = per_cut * importance
    return per_cut.sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# Soft Quadratic Weighted Kappa loss  ( = 1 - kappa,  differentiable )        #
# --------------------------------------------------------------------------- #
def qwk_loss(logits: torch.Tensor, target: torch.Tensor,
             num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)                          # (B, K)
    dev = logits.device
    idx = torch.arange(num_classes, device=dev).float()
    W = (idx.view(-1, 1) - idx.view(1, -1)) ** 2 / (num_classes - 1) ** 2   # (K, K)

    t_oh = F.one_hot(target, num_classes).float()             # (B, K)
    O = t_oh.t() @ probs                                      # observed  (K, K)
    hist_t = t_oh.sum(0)
    hist_p = probs.sum(0)
    E = torch.outer(hist_t, hist_p) / target.size(0)          # expected  (K, K)

    return (W * O).sum() / ((W * E).sum() + eps)              # 1 - kappa


# --------------------------------------------------------------------------- #
# Combined multi-task loss                                                    #
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any], class_weight: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = len(cfg["classes"])
        w = cfg["train"]["loss_weights"]
        self.w = {k: float(w.get(k, 0.0)) for k in ("stage", "ordinal", "referable", "qwk")}
        self.focal = FocalLoss(cfg["train"].get("focal_gamma", 2.0), class_weight)

    def _compute(self, out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor]):
        l_stage = self.focal(out["stage"], tgt["stage"])
        levels = tgt.get("ordinal")
        if levels is None:
            levels = stage_to_levels(tgt["stage"], self.num_classes)
        l_ord = coral_loss(out["ordinal"], levels)
        l_ref = F.binary_cross_entropy_with_logits(out["referable"], tgt["referable"].float())
        l_qwk = qwk_loss(out["stage"], tgt["stage"], self.num_classes)
        total = (self.w["stage"] * l_stage + self.w["ordinal"] * l_ord
                 + self.w["referable"] * l_ref + self.w["qwk"] * l_qwk)
        parts = {"stage": l_stage.detach(), "ordinal": l_ord.detach(),
                 "referable": l_ref.detach(), "qwk": l_qwk.detach()}
        return total, parts

    def forward(self, out, tgt, targets_b: dict | None = None, lam: float = 1.0):
        total, parts = self._compute(out, tgt)
        if targets_b is not None and lam < 1.0:
            total_b, _ = self._compute(out, targets_b)
            total = lam * total + (1.0 - lam) * total_b
        parts["total"] = total.detach()
        return total, parts


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.losses`                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config

    cfg = load_config()
    K = len(cfg["classes"])
    B = 8
    out = {"stage": torch.randn(B, K, requires_grad=True),
           "ordinal": torch.randn(B, K - 1, requires_grad=True),
           "referable": torch.randn(B, requires_grad=True)}
    tgt = {"stage": torch.randint(0, K, (B,)),
           "referable": torch.randint(0, 2, (B,)).float()}
    tgt["ordinal"] = stage_to_levels(tgt["stage"], K)

    crit = MultiTaskLoss(cfg, class_weight=torch.tensor([0.2, 1.1, 0.4, 2.0, 1.3]))
    loss, parts = crit(out, tgt)
    loss.backward()
    print("loss:", float(loss.detach()))
    print("parts:", {k: round(float(v), 4) for k, v in parts.items()})
    print("grad on stage logits:", out["stage"].grad is not None)
