# DR-Triage prototype app

Gradio interface over the full pipeline: quality gate → preprocess → grade →
Grad-CAM++ → RetinaBot, plus a batch-triage tab.

## Run

```bash
cd dr-triage
source .venv/bin/activate
python app/app.py
```

Opens on `http://127.0.0.1:7860`. For a public link (e.g. from Colab):

```bash
GRADIO_SHARE=1 python app/app.py
```

Requires `outputs/checkpoints/fold*_best.pt` (the M5 run). Without them the gate
and RetinaBot still work but grading is disabled.

## Tabs

| Tab | What it does |
|-----|--------------|
| **Grade an image** | upload → quality gate (reject with reason, or) preprocessed preview + grade + referable decision + calibrated confidence + Grad-CAM++ overlay; seeds the chat with an explanation |
| **Ask RetinaBot** | guideline-grounded Q&A, using the last grade as context |
| **Batch triage** | score many images at once → severity-sorted table + CSV download |

## Screenshots for the report (§8.2)

Capture three:
1. an **accepted** image showing the grade, Grad-CAM overlay, confidence and action;
2. a **rejected** low-quality image showing the reason(s) — drag any fundus image
   through a blur/darken first, or use a non-fundus photo;
3. a **RetinaBot** exchange (the seeded explanation + one follow-up question).

Save them to `outputs/figures/08_app_*.png` and reference them in `report/REPORT.md`.
