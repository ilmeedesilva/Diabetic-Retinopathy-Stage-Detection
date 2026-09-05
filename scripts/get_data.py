"""
Download the APTOS 2019 Blindness Detection dataset for local development.
===================================================================

Two ways to get the data locally; this script tries them in order.

A. Kaggle API (official, ~9 GB — full-resolution images)
   1. pip install kaggle
   2. Kaggle -> account -> "Create New API Token" -> saves kaggle.json
   3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   4. Accept the competition rules (competition page -> "Late Submission" -> "I Understand and Accept")
   5. python scripts/get_data.py

B. Manual fallback
   Download `aptos2019-blindness-detection.zip` from the competition "Data" tab,
   unzip it into  data/aptos2019/  so that these exist:
       data/aptos2019/train.csv
       data/aptos2019/train_images/*.png

For fast iteration on a laptop you can also use a pre-resized community mirror
(search Kaggle Datasets for "APTOS 2019 512x512"); if you do, point
paths.local_root in config.yaml at that folder and keep labels_csv/image_subdir
matching its layout.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data" / "aptos2019"
COMPETITION = "aptos2019-blindness-detection"


def already_present() -> bool:
    return (DEST / "train.csv").is_file() and (DEST / "train_images").is_dir()


def download_with_kaggle_cli() -> bool:
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading competition '{COMPETITION}' via Kaggle CLI -> {DEST}")
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DEST)],
            check=True,
        )
    except FileNotFoundError:
        print("  'kaggle' CLI not found. Run: pip install kaggle")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  Kaggle CLI failed (rc={e.returncode}). Have you accepted the "
              f"competition rules and installed kaggle.json?")
        return False

    zips = sorted(DEST.glob("*.zip"))
    for z in zips:
        print(f"  Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DEST)
        z.unlink()

    # Some Kaggle bundles nest images in train_images.zip
    inner = DEST / "train_images.zip"
    if inner.is_file():
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(DEST / "train_images")
        inner.unlink()
    return already_present()


def main() -> int:
    if already_present():
        n = len(list((DEST / "train_images").glob("*")))
        print(f"Data already present at {DEST}  ({n} images). Nothing to do.")
        return 0

    if download_with_kaggle_cli() and already_present():
        n = len(list((DEST / "train_images").glob("*")))
        print(f"\nDone. {n} training images at {DEST}")
        print("You can now run notebooks/01_eda_split.ipynb locally.")
        return 0

    print("\nAutomatic download did not complete. Follow option B in this file's "
          "docstring (manual download into data/aptos2019/).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
