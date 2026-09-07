# DR-Triage — Explainable, Order-Aware Diabetic Retinopathy Screening

Computer Vision coursework (BSc Hons Computer Science, NIBM).
**Task:** classify diabetic retinopathy (DR) *and* its 5-point severity stage from
retinal fundus photographs, using image preprocessing, augmentation, and a CNN with
transfer learning — then wrap the model in a small explainable screening-triage tool.

## What makes this different from a typical submission

| Typical submission | This project |
|---|---|
| Plain 5-class softmax | **Ordinal (CORN) head + quadratic-weighted-kappa-aware loss** — stages are ordered, so grading *No DR* as *Proliferative* is punished harder than grading *Moderate* as *Severe* |
| Resize + histogram equalisation | **Circle-crop → green-channel → CLAHE → Ben Graham normalisation** (the 2015 Kaggle DR winner's technique) |
| Predict the class, stop | **Referable-DR decision** (stages 2–4) + **calibrated confidence** + **MC-Dropout uncertainty** flag |
| Feed any image to the model | **Image-quality gate** rejects blurred / poorly-lit / off-centre photos before prediction |
| "Trust the accuracy" | **Grad-CAM++ explanations** + **RetinaBot**, a retrieval-grounded chatbot that answers DR questions from cited screening-guideline snippets |

## Pipeline

```
fundus image
   -> quality gate (blur / brightness / field-of-view)
   -> preprocess (crop, green channel, CLAHE, Ben Graham, resize, normalise)
   -> EfficientNetV2-S backbone (ImageNet transfer learning) + CBAM attention
   -> shared feature vector
        -> stage head        (5-class softmax)
        -> ordinal head       (CORN, order-aware)
        -> referable head     (binary: refer to ophthalmologist?)
   -> temperature scaling (calibrated probabilities) + MC-Dropout (uncertainty)
   -> Grad-CAM++ overlay + RetinaBot explanation
```

## Dataset

**APTOS 2019 Blindness Detection** (Kaggle) — 3,662 clinician-graded fundus images,
5 stages: `0` No DR, `1` Mild, `2` Moderate, `3` Severe, `4` Proliferative DR.
Chosen for native stage labels, a GPU-friendly size, and a realistic class imbalance
that lets the balancing techniques be demonstrated rather than just described.

## Repo layout

```
dr-triage/
├── config.yaml                     # single source of truth for paths / hyperparams
├── requirements.txt
├── src/
│   ├── data.py                     # M1  loading, stratified split, PyTorch Dataset
│   ├── preprocess.py               # M2  crop / green channel / CLAHE / Ben Graham
│   ├── augment.py                  # M3  albumentations + mixup / cutmix
│   ├── model.py                    # M4  EfficientNetV2 + CBAM + 3 heads
│   ├── losses.py                   # M4  focal + CORN ordinal + QWK-aware
│   ├── train.py                    # M5  2-phase training, AMP, cosine, CV
│   ├── evaluate.py                 # M6  metrics, Grad-CAM, calibration, uncertainty
│   ├── quality_gate.py             # M7  ungradable-image rejection
│   └── chatbot.py                  # M7  FAISS retrieval + cited answers
├── notebooks/
│   ├── 01_eda_split.ipynb          # M1  EDA + stratified split
│   ├── 02_preprocess.ipynb         # M2  preprocessing pipeline
│   ├── 03_augment.ipynb            # M3  augmentation + class balancing
│   ├── 04_model.ipynb              # M4  architecture walkthrough + losses
│   ├── 05_train.ipynb              # M5  smoke test here; full run needs GPU (Kaggle/Colab)
│   ├── 06_evaluate.ipynb           # M6
│   └── 07_app_demo.ipynb           # M7-M8
├── app/app.py                      # Gradio demo
├── kb/                             # DR guideline snippets + sources
└── outputs/                        # splits, figures, checkpoints, metrics
```

## Milestones

| # | Milestone | Status |
|---|---|---|
| M1 | Data understanding, EDA, stratified split | done |
| M2 | Preprocessing pipeline (crop, FOV mask, Ben Graham, CLAHE) | done |
| M3 | Augmentation + class balancing (Albumentations, 3 strategies, MixUp/CutMix) | done |
| M4 | Model (EfficientNetV2-S + CBAM + stage/ordinal/referable heads) + losses | done |
| M5 | Two-phase training + 3-fold stratified CV | done (smoke-tested; full GPU run pending) |
| M6 | Evaluation (curves, P/R/F1, QWK, ROC/PR, calibration, MC-dropout, Grad-CAM++, error analysis) | done — test QWK 0.854, referable AUC 0.979 |
| M7 | Image-quality gate + RetinaBot (grounded RAG chatbot) | done |
| M8 | Gradio app (`app/app.py`) + report draft | app done; screenshots + video + PDF export are manual |

## Running it

Paths auto-resolve via `src.data.resolve_data_paths()`:
`$DR_DATA_ROOT` → `paths.local_root` → `paths.kaggle_root`. The same notebooks run
in both places with no edits.

### Locally, in VS Code (recommended for M1–M3, M6–M8)

```bash
cd dr-triage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_data.py          # needs kaggle.json + accepted comp rules
```

Then open `notebooks/01_eda_split.ipynb`, pick the `.venv` kernel, Run All.
Training (M5) also runs on an Apple-Silicon Mac via the PyTorch **MPS** backend,
just slower — expect ~30–60 min per CV fold.

### On Kaggle (recommended for M5 training — free GPU)

1. Accept the *APTOS 2019 Blindness Detection* competition rules
   (competition page → **Late Submission** → **I Understand and Accept**).
2. New Notebook → **Add Input** → *Competitions* → APTOS 2019.
3. **Add Input → Upload** `dr-triage.zip`; name it `dr-triage`.
4. `File → Import Notebook` → the `.ipynb` you want → **Run All**.

All randomness is seeded via `config.yaml: seed` (default 42).
