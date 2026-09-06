"""
M1 — Data understanding, stratified splitting, and the PyTorch Dataset.
===================================================================

Responsibilities of this module:

1. `load_config`      : read `config.yaml` once, share it everywhere.
2. `set_seed`         : make numpy / random / torch deterministic.
3. `load_labels`      : read the APTOS `train.csv`, attach file paths, and derive
                        the extra targets our multi-task model needs
                        (ordinal encoding + the binary "referable" flag).
4. `make_splits`      : ONE stratified train / val / test split, written to disk so
                        every later notebook reads the exact same rows
                        (this is how we prevent data leakage between stages).
5. `split_summary`    : a table proving each split keeps the class proportions.
6. `stratified_folds` : stratified K-fold indices on the TRAIN portion for CV (M5).
7. `class_weights`    : inverse-frequency weights for the imbalance-aware loss.
8. `APTOSDataset`     : returns (image_tensor, targets_dict) for the DataLoader.

Design choices worth noting for the report
------------------------------------------
* The split is created **before any augmentation or preprocessing statistics are
  computed**, and the test set is written out separately and never read during
  training — this is the leakage guard the rubric asks for.
* Stages are ordered (0 < 1 < 2 < 3 < 4). We keep the raw class label for the
  softmax head but also precompute a CORN-style ordinal target so the model can
  learn the ordering (M4).
* "Referable DR" = stage >= 2 : the single most clinically important decision,
  exposed as its own head and its own evaluation later.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# torch is only needed for the Dataset class; keep the import soft so the pure
# EDA parts of notebook 01 can run even in a no-torch kernel.
try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


# --------------------------------------------------------------------------- #
# 1. Config                                                                   #
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load `config.yaml`. Searches the given path, then one directory up
    (so it works whether you run from repo root or from `notebooks/`)."""
    for candidate in (Path(path), Path("..") / path, Path(__file__).resolve().parents[1] / "config.yaml"):
        if candidate.is_file():
            with open(candidate) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(f"config.yaml not found (looked near {path!r})")


# --------------------------------------------------------------------------- #
# 2. Reproducibility                                                          #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed every RNG we touch so splits and training are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# 3. Labels                                                                   #
# --------------------------------------------------------------------------- #
def _ordinal_target(stage: int, num_classes: int = 5) -> list[int]:
    """CORN / CORAL style encoding: stage k -> [1]*k + [0]*(K-1-k).

    stage 0 -> [0, 0, 0, 0]      stage 3 -> [1, 1, 1, 0]
    stage 2 -> [1, 1, 0, 0]      stage 4 -> [1, 1, 1, 1]

    Each position i answers a binary question "is the stage > i ?", which is what
    lets the model exploit the ordering instead of treating classes as unrelated.
    """
    return [1 if stage > i else 0 for i in range(num_classes - 1)]


def resolve_data_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """Pick the dataset root that actually exists on this machine.

    Order: $DR_DATA_ROOT  ->  paths.local_root  ->  paths.kaggle_root  ->
    *any* attached Kaggle dataset that contains labels_csv. That last step means
    you do NOT need to edit config.yaml on Kaggle — just attach the same dataset
    (Add Input) and this finds it automatically, exactly like the notebooks'
    setup cell auto-finds the uploaded src/ code.
    """
    import glob

    p = cfg["paths"]
    csv_name = p.get("labels_csv", "train.csv")
    img_name = p.get("image_subdir", "train_images")

    def is_valid_root(root: str) -> bool:
        """train.csv existing is not enough — e.g. the raw APTOS competition also
        has a train.csv, but no `colored_images/` folder. Require the expected
        image layout to actually be present too."""
        if not root or not (Path(root) / csv_name).is_file():
            return False
        img_dir = Path(root) / img_name
        if not img_dir.is_dir():
            return False
        if p.get("image_layout", "flat") == "nested":
            class_dirs = p.get("class_dirs", {})
            return any((img_dir / d).is_dir() for d in class_dirs.values())
        return True

    candidates = []
    if os.environ.get("DR_DATA_ROOT"):
        candidates.append(os.environ["DR_DATA_ROOT"])
    candidates += [p.get("local_root"), p.get("kaggle_root")]

    for root in candidates:
        if is_valid_root(root):
            return {
                "root": str(root),
                "train_csv": str(Path(root) / csv_name),
                "train_images": str(Path(root) / img_name),
            }

    # Fallback: scan every attached Kaggle dataset for labels_csv, at ANY depth —
    # classic UI mounts at /kaggle/input/<dataset>/..., the newer UI namespaces by
    # owner at /kaggle/input/datasets/<owner>/<dataset>/... — recursive covers both.
    # is_valid_root rejects false positives (e.g. a `train.csv` shipped inside
    # outputs/splits/ of the uploaded code bundle has no matching image folder).
    hits = [h for h in glob.glob(f"/kaggle/input/**/{csv_name}", recursive=True)
           if is_valid_root(str(Path(h).parent))]
    if hits:
        root = str(Path(hits[0]).parent)
        return {"root": root, "train_csv": str(Path(root) / csv_name),
                "train_images": str(Path(root) / img_name)}

    # Nothing found — return the Kaggle guess so the error message is informative.
    root = p.get("kaggle_root", "")
    return {
        "root": root,
        "train_csv": str(Path(root) / csv_name),
        "train_images": str(Path(root) / img_name),
    }


def load_labels(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read train.csv and enrich it with everything downstream code needs."""
    dp = resolve_data_paths(cfg)
    csv_path = Path(dp["train_csv"])
    img_dir = Path(dp["train_images"])
    ext = cfg["paths"]["image_ext"]
    ref_thr = cfg["referable_threshold"]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found. Set $DR_DATA_ROOT, or download the data with "
            f"`python scripts/get_data.py`, or fix paths.local_root in config.yaml."
        )
    print(f"[load_labels] data root: {dp['root']}")

    df = pd.read_csv(csv_path)
    # APTOS columns: id_code (filename stem), diagnosis (0-4)
    df = df.rename(columns={"diagnosis": "stage"})
    df["stage"] = df["stage"].astype(int)
    df["stage_name"] = df["stage"].map(cfg["classes"])
    df["referable"] = (df["stage"] >= ref_thr).astype(int)
    df["ordinal"] = df["stage"].map(lambda k: _ordinal_target(k, len(cfg["classes"])))
    df = attach_image_paths(df, cfg)

    missing = [p for p in df["path"].head(50) if not Path(p).exists()]
    if missing:
        print(f"[load_labels] WARNING: {len(missing)}/50 sampled image files not "
              f"found — check paths.image_subdir / image_layout in config.yaml. "
              f"First missing: {missing[0]}")
    return df


def attach_image_paths(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """(Re)compute the `path` column for whatever machine this is running on.

    Split CSVs are portable across machines (they only need id_code + stage), but
    a `path` baked in on one machine (e.g. your laptop) is wrong on another (e.g.
    Kaggle) — so every loader calls this instead of trusting a stored path.

    Two on-disk layouts, selected by paths.image_layout:
        "flat"   -> <img_dir>/<id_code>.png            (raw Kaggle competition)
        "nested" -> <img_dir>/<class_dir>/<id_code>.png (many resized mirrors,
                    e.g. colored_images/{No_DR,Mild,Moderate,Severe,Proliferate_DR})
    """
    dp = resolve_data_paths(cfg)
    img_dir = Path(dp["train_images"])
    ext = cfg["paths"]["image_ext"]
    df = df.copy()
    if cfg["paths"].get("image_layout", "flat") == "nested":
        class_dirs = cfg["paths"]["class_dirs"]        # {stage_int: folder_name}
        df["path"] = df.apply(
            lambda r: str(img_dir / class_dirs[r["stage"]] / f"{r['id_code']}{ext}"),
            axis=1,
        )
    else:
        df["path"] = df["id_code"].map(lambda s: str(img_dir / f"{s}{ext}"))
    return df


# --------------------------------------------------------------------------- #
# 4. Stratified split (train / val / test)                                    #
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame, cfg: dict[str, Any], save: bool = True) -> dict[str, pd.DataFrame]:
    """Single stratified split into train / val / test.

    Returns {"train": df, "val": df, "test": df} and (by default) writes each to
    `outputs/splits/{name}.csv`. Later notebooks load those CSVs rather than
    re-splitting, so the test set is genuinely held out.
    """
    from sklearn.model_selection import train_test_split

    seed = cfg["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    strat_col = cfg["split"]["stratify_on"]
    strat_col = "stage" if strat_col == "diagnosis" else strat_col

    # First carve off the test set...
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[strat_col], random_state=seed
    )
    # ...then split the remainder so val is `val_size` of the ORIGINAL dataset.
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val[strat_col], random_state=seed
    )

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    if save:
        out = Path(cfg["paths"]["splits_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            # `ordinal` is a list column -> store as a plain string, re-parse on load
            part.assign(ordinal=part["ordinal"].map(lambda v: " ".join(map(str, v)))) \
                .to_csv(out / f"{name}.csv", index=False)
        print(f"[make_splits] wrote {', '.join(f'{k}={len(v)}' for k, v in splits.items())} "
              f"to {out}/")
    return splits


def load_splits(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Re-load the split CSVs written by `make_splits`, restoring list columns and
    recomputing `path` for the current machine (see `attach_image_paths`) — this
    is what makes the same split CSVs work unchanged on a laptop and on Kaggle."""
    out = Path(cfg["paths"]["splits_dir"])
    splits = {}
    for name in ("train", "val", "test"):
        part = pd.read_csv(out / f"{name}.csv")
        part["ordinal"] = part["ordinal"].map(lambda s: [int(x) for x in str(s).split()])
        part = attach_image_paths(part, cfg)
        splits[name] = part
    return splits


def split_summary(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-split class counts and percentages — evidence that stratification held."""
    rows = []
    for name, part in splits.items():
        counts = part["stage"].value_counts().sort_index()
        pct = (counts / len(part) * 100).round(1)
        for stage in cfg["classes"]:
            rows.append({
                "split": name,
                "stage": stage,
                "stage_name": cfg["classes"][stage],
                "count": int(counts.get(stage, 0)),
                "pct": float(pct.get(stage, 0.0)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Stratified K-fold on the TRAIN portion (used in M5)                      #
# --------------------------------------------------------------------------- #
def stratified_folds(train_df: pd.DataFrame, cfg: dict[str, Any]):
    """Yield (fold_idx, train_index, val_index) for cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg["split"]["n_folds"], shuffle=True,
                          random_state=cfg["seed"])
    for i, (tr, va) in enumerate(skf.split(train_df, train_df["stage"])):
        yield i, tr, va


# --------------------------------------------------------------------------- #
# 6. Class weights for the imbalance-aware loss                               #
# --------------------------------------------------------------------------- #
def class_weights(train_df: pd.DataFrame, num_classes: int = 5) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.0."""
    counts = train_df["stage"].value_counts().reindex(range(num_classes), fill_value=0)
    inv = counts.sum() / (num_classes * counts.clip(lower=1))
    return (inv / inv.mean()).to_numpy(dtype="float32")


# --------------------------------------------------------------------------- #
# 7. PyTorch Dataset                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    image_size: int = 224
    preprocess: Any = None      # callable(np.uint8 HxWx3 RGB) -> np.uint8, from M2
    transform: Any = None       # albumentations Compose, from M3
    return_path: bool = False


class APTOSDataset(Dataset):
    """Wraps a split DataFrame. Each item is (image, targets) where targets is a
    dict with keys: stage (long), ordinal (float[K-1]), referable (float)."""

    def __init__(self, df: pd.DataFrame, dcfg: DatasetConfig):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to instantiate APTOSDataset.")
        self.df = df.reset_index(drop=True)
        self.dcfg = dcfg

    def __len__(self) -> int:
        return len(self.df)

    def _read_rgb(self, path: str) -> np.ndarray:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._read_rgb(row["path"])

        # M2 preprocessing (crop / green channel / CLAHE / Ben Graham) if provided
        if self.dcfg.preprocess is not None:
            img = self.dcfg.preprocess(img)

        # M3 augmentation + tensor conversion if provided; otherwise a plain resize
        if self.dcfg.transform is not None:
            img = self.dcfg.transform(image=img)["image"]
        else:
            import cv2
            img = cv2.resize(img, (self.dcfg.image_size, self.dcfg.image_size))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        targets = {
            "stage": torch.tensor(int(row["stage"]), dtype=torch.long),
            "ordinal": torch.tensor(row["ordinal"], dtype=torch.float32),
            "referable": torch.tensor(float(row["referable"]), dtype=torch.float32),
        }
        if self.dcfg.return_path:
            return img, targets, row["path"]
        return img, targets


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.data`                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    print("Config loaded. Classes:", cfg["classes"])
    try:
        df = load_labels(cfg)
        print(f"Loaded {len(df)} rows. Stage distribution:")
        print(df["stage"].value_counts().sort_index())
        splits = make_splits(df, cfg, save=False)
        print(split_summary(splits, cfg).to_string(index=False))
        print("Class weights:", class_weights(splits["train"]))
    except FileNotFoundError as e:
        print(f"(No dataset locally — expected on your laptop.) {e}")
