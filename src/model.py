"""
M4 — Model: EfficientNetV2-S backbone + CBAM attention + three task heads.
===================================================================

Surface-level architecture
--------------------------

    input  x : (B, 3, S, S)         preprocessed fundus image, S = 224 or 380
      │
      ▼
    EfficientNetV2-S backbone       ImageNet-21k -> 1k pretrained (timm)
    (transfer learning)             compound-scaled MBConv / Fused-MBConv blocks
      │  feature map (B, 1280, S/32, S/32)
      ▼
    CBAM attention                  channel attention  ×  spatial attention
      │  re-weighted feature map (same shape)          -> focuses on lesion regions
      ▼
    Global Average Pool + Dropout
      │  embedding v : (B, 1280)
      ├───────────────► stage head      Linear(1280, 5)   softmax   -> 5-way severity
      ├───────────────► ordinal head    CORAL: shared Linear(1280,1) + 4 biases
      │                                  -> 4 rank-monotonic logits  -> ordered stage
      └───────────────► referable head  Linear(1280, 1)   sigmoid   -> refer? (stage>=2)

Why this design (expanded in notebook 04):
* EfficientNetV2-S — better accuracy-per-parameter and much faster training than
  ResNet-50 / VGG-16 at similar accuracy; fits a single 16 GB GPU at 380 px.
* CBAM — a light (<0.2 M param) attention module that lets the network suppress
  the uniform background and amplify small bright/dark lesions; also gives a
  cleaner Grad-CAM in M6.
* Multi-task — the stage grade, its *ordering*, and the clinical refer/no-refer
  decision are related targets; sharing a backbone regularises all three.
* Ordinal (CORAL) head — DR stages are ordered, so predicting "No DR" when the
  truth is "Proliferative" must cost more than predicting "Moderate". CORAL
  (Cao et al. 2020) produces rank-consistent probabilities by construction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018)         #
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    """Which feature *channels* matter — a shared MLP over avg- and max-pooled
    channel descriptors."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        return torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Which *locations* matter — a conv over the channel-wise avg and max maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
        self._last_spatial: torch.Tensor | None = None  # cached for visualisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        s = self.sa(x)
        self._last_spatial = s.detach()
        return x * s


# --------------------------------------------------------------------------- #
# The full model                                                             #
# --------------------------------------------------------------------------- #
class DRModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], pretrained: bool | None = None):
        super().__init__()
        import timm

        mcfg = cfg["model"]
        self.num_classes = len(cfg["classes"])
        name = mcfg["backbone"]
        if pretrained is None:
            pretrained = mcfg.get("pretrained", True)

        # num_classes=0 + global_pool='' -> backbone returns the raw feature map
        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features

        self.use_cbam = mcfg.get("attention", "cbam") == "cbam"
        self.cbam = CBAM(feat_dim) if self.use_cbam else nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(mcfg.get("dropout", 0.3))

        # heads
        self.head_stage = nn.Linear(feat_dim, self.num_classes)              # softmax / CE
        self.coral_fc = nn.Linear(feat_dim, 1, bias=False)                   # CORAL shared weight
        self.coral_bias = nn.Parameter(torch.zeros(self.num_classes - 1))    # CORAL per-cut bias
        self.head_referable = nn.Linear(feat_dim, 1)                         # sigmoid / BCE

        self.feat_dim = feat_dim

    # -- forward pieces ---------------------------------------------------- #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + CBAM -> (B, C, h, w). Used for Grad-CAM (M6) and the
        feature-map visualisation in notebook 04."""
        f = self.backbone(x)
        f = self.cbam(f)
        return f

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        f = self.forward_features(x)
        v = self.dropout(self.pool(f).flatten(1))                # (B, C)
        out = {
            "stage": self.head_stage(v),                         # (B, 5)
            "ordinal": self.coral_fc(v) + self.coral_bias,       # (B, 4) rank-monotonic
            "referable": self.head_referable(v).squeeze(1),      # (B,)
        }
        if return_features:
            out["features"] = f
        return out

    # -- convenience ----------------------------------------------------- #
    @property
    def cam_target_layer(self) -> nn.Module:
        """Last spatial layer for Grad-CAM: the CBAM output (or backbone if no CBAM)."""
        return self.cbam if self.use_cbam else self.backbone

    def group_parameters(self):
        """(backbone_params, head_params) for discriminative learning rates in M5."""
        head_mods = [self.cbam, self.head_stage, self.coral_fc, self.head_referable]
        head_ids = {id(p) for m in head_mods for p in m.parameters()}
        head_ids.add(id(self.coral_bias))
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        heads = [p for p in self.parameters() if id(p) in head_ids]
        return backbone, heads


# --------------------------------------------------------------------------- #
# Prediction helpers                                                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_stage(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Turn raw head outputs into stage predictions.

    * softmax_pred : argmax of the 5-way head (per-class metrics, Grad-CAM)
    * ordinal_pred : number of CORAL cuts passed (order-aware; primary metric)
    * referable_prob: sigmoid of the referable head
    """
    soft = out["stage"].softmax(dim=1)
    ordinal_prob = out["ordinal"].sigmoid()
    return {
        "softmax_pred": soft.argmax(dim=1),
        "softmax_prob": soft,
        "ordinal_pred": (ordinal_prob > 0.5).sum(dim=1),
        "ordinal_prob": ordinal_prob,
        "referable_prob": out["referable"].sigmoid(),
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {"total": total, "trainable": trainable}
    if isinstance(model, DRModel):
        bb, hd = model.group_parameters()
        parts["backbone"] = sum(p.numel() for p in bb)
        parts["heads_and_cbam"] = sum(p.numel() for p in hd)
    return parts


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.model`                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.data import load_config

    cfg = load_config()
    try:
        model = DRModel(cfg, pretrained=True)
        src = "pretrained"
    except Exception as e:  # offline / no weights cache
        print(f"(pretrained weights unavailable: {e}); using random init")
        model = DRModel(cfg, pretrained=False)
        src = "random"
    model.eval()

    s = cfg["image"]["size_stage1"]
    x = torch.randn(2, 3, s, s)
    out = model(x, return_features=True)
    print(f"backbone: {cfg['model']['backbone']}  ({src})  feat_dim={model.feat_dim}")
    print("feature map :", tuple(out["features"].shape))
    for k in ("stage", "ordinal", "referable"):
        print(f"{k:10s}: {tuple(out[k].shape)}")
    print("params      :", {k: f"{v/1e6:.2f}M" for k, v in count_parameters(model).items()})
    pred = predict_stage(out)
    print("softmax_pred:", pred["softmax_pred"].tolist(),
          "| ordinal_pred:", pred["ordinal_pred"].tolist())
