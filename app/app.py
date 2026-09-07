"""
M8 — DR-Triage prototype (Gradio).
===================================================================

The whole pipeline in one interface:

    upload fundus image
      -> image-quality gate (reject with reason, or)
      -> preprocessing preview
      -> stage grade + referable-DR decision + calibrated confidence
      -> Grad-CAM++ overlay (where the model looked)
      -> RetinaBot Q&A, seeded with this result as context

plus a Batch-triage tab that scores a folder of images and returns a CSV sorted
by severity ("screening clinic" mode).

Run:
    python app/app.py                 # local, http://127.0.0.1:7860
    GRADIO_SHARE=1 python app/app.py   # public share link (Colab)

Needs the trained checkpoints in outputs/checkpoints/ (fold*_best.pt). Without
them the gate and RetinaBot still work; grading is disabled.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr

from src.data import load_config
from src.quality_gate import QualityGate
from src.preprocess import make_preprocessor
from src.augment import build_eval_aug
from src.model import predict_stage
from src import evaluate as E
from src.chatbot import RetinaBot

# --------------------------------------------------------------------------- #
# Load everything once                                                        #
# --------------------------------------------------------------------------- #
CFG = load_config()
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = CFG["classes"]
S = CFG["image"]["size_stage2"]
MEAN, STD = CFG["image"]["mean"], CFG["image"]["std"]
T = float(CFG.get("inference", {}).get("temperature", 1.0))
REF_THR = float(CFG.get("inference", {}).get("referable_threshold", 0.5))

print(f"[app] device={DEV}  temperature={T}  referable_threshold={REF_THR}")
GATE = QualityGate.from_config(CFG)
PRE = make_preprocessor(CFG)
EVAL_TF = build_eval_aug(S, MEAN, STD)
try:
    MODELS = E.load_fold_models(CFG, DEV)
except FileNotFoundError:
    MODELS = []
    print("[app] WARNING: no checkpoints in outputs/checkpoints/ — grading disabled")
BOT = RetinaBot(kb_path=CFG["chatbot"]["kb_path"], model_name=CFG["chatbot"]["encoder"],
                min_score=CFG["chatbot"]["min_score"], top_k=CFG["chatbot"]["top_k"])
print(f"[app] ensemble={len(MODELS)}  kb_snippets={len(BOT.snippets)}")


# --------------------------------------------------------------------------- #
# Inference helpers                                                           #
# --------------------------------------------------------------------------- #
def _predict(img_rgb: np.ndarray):
    x = EVAL_TF(image=PRE(img_rgb))["image"].unsqueeze(0).to(DEV)
    logit = ordp = refp = 0.0
    with torch.no_grad():
        for m in MODELS:
            o = m(x)
            logit = logit + o["stage"].float()
            ordp = ordp + o["ordinal"].sigmoid().float()
            refp = refp + o["referable"].sigmoid().float()
    n = max(len(MODELS), 1)
    stage_probs = torch.softmax(logit / n / T, dim=1)[0].cpu().numpy()
    ord_probs = (ordp / n)[0].cpu().numpy()
    stage = int((ord_probs > 0.5).sum())
    return stage, stage_probs, float((refp / n)[0])


def _overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_r = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Callbacks                                                                   #
# --------------------------------------------------------------------------- #
def grade(img):
    if img is None:
        return "Upload a fundus image and press **Assess**.", None, None, None, []

    rep = GATE.assess(img)
    m = rep.metrics
    stats = (f"<sub>sharpness {m['sharpness']:.0f} · brightness {m['brightness']:.0f} · "
             f"contrast {m['contrast']:.3f} · field-of-view {m['fov_coverage']:.2f}</sub>")
    if not rep.passed:
        msg = "### ⛔ Image rejected before grading\n\n" + \
              "\n".join(f"- {r}" for r in rep.reasons) + "\n\n" + stats
        return msg, None, None, None, []

    proc = PRE(img)
    if not MODELS:
        return "### ✅ Quality OK\n\n_No trained checkpoints found, so grading is disabled._\n\n" + stats, proc, None, None, []

    stage, sp, refp = _predict(img)
    conf = float(sp[stage])
    refer = refp >= REF_THR
    cam = E.gradcam_plus_plus(MODELS[0], EVAL_TF(image=proc)["image"].unsqueeze(0), [stage], DEV)[0]
    overlay = _overlay(proc, cam)

    bars = "<br>".join(
        f"{CLASSES[i]}: <b>{sp[i]:.0%}</b>" + (" ◀" if i == stage else "") for i in range(len(CLASSES)))
    md = (f"## Grade {stage} — {CLASSES[stage]}\n\n"
          f"| | |\n|---|---|\n"
          f"| Calibrated confidence | **{conf:.0%}** |\n"
          f"| Referable-DR probability | **{refp:.0%}** |\n"
          f"| Recommended action | **{'REFER to ophthalmology' if refer else 'Routine re-screen'}** |\n\n"
          f"{bars}\n\n{stats}\n\n"
          f"<sub>Screening aid — not a diagnosis. A clinician makes the final decision.</sub>")

    ctx = {"stage": stage, "confidence": conf, "referable_prob": refp}
    intro = BOT.answer("What does this result mean for me?", result_context=ctx)["answer"]
    chat = [{"role": "assistant", "content": intro}]
    return md, proc, overlay, ctx, chat


def chat_fn(message, history, ctx):
    if not message:
        return history, ""
    out = BOT.answer(message, result_context=ctx)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": out["answer"]},
    ]
    return history, ""


def batch(files):
    rows = []
    for f in files or []:
        name = Path(f).name
        img = cv2.imread(f)
        if img is None:
            rows.append([name, "unreadable", "", "", "", ""]); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rep = GATE.assess(img)
        if not rep.passed:
            rows.append([name, "rejected", "", rep.reasons[0][:48], "", "retake image"]); continue
        if not MODELS:
            rows.append([name, "no model", "", "", "", ""]); continue
        stage, sp, refp = _predict(img)
        rows.append([name, "graded", stage, CLASSES[stage], round(refp, 2),
                     "REFER" if refp >= REF_THR else "re-screen"])
    df = pd.DataFrame(rows, columns=["file", "status", "stage", "grade", "referable_prob", "action"])
    df["_s"] = pd.to_numeric(df["stage"], errors="coerce").fillna(-1)
    df = df.sort_values(["_s", "referable_prob"], ascending=False).drop(columns="_s").reset_index(drop=True)
    out_csv = Path(tempfile.gettempdir()) / "dr_triage_batch.csv"
    df.to_csv(out_csv, index=False)
    return df, str(out_csv)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
with gr.Blocks(title="DR-Triage") as demo:
    gr.Markdown("# DR-Triage — Diabetic Retinopathy Screening Assistant\n"
                "Explainable, order-aware DR staging with a referral decision, an "
                "image-quality gate and a guideline-grounded assistant. "
                "**Screening aid, not a diagnosis.**")
    ctx_state = gr.State(None)

    with gr.Tab("Grade an image"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="Fundus image", height=340)
            with gr.Column():
                proc_img = gr.Image(label="Preprocessed", height=160)
                cam_img = gr.Image(label="Grad-CAM++ — where the model looked", height=160)
        assess_btn = gr.Button("Assess", variant="primary")
        result_md = gr.Markdown()

    with gr.Tab("Ask RetinaBot"):
        gr.Markdown("_Grade an image first and the assistant starts with an "
                    "explanation of that result. Answers are drawn only from cited "
                    "screening-guidance snippets._")
        chatbot = gr.Chatbot(height=380, label="RetinaBot")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask about your result or diabetic retinopathy…",
                             scale=5, show_label=False)
            send = gr.Button("Send", scale=1)
        clear = gr.Button("Clear chat")

    with gr.Tab("Batch triage"):
        gr.Markdown("Upload multiple fundus images; get a severity-sorted triage "
                    "list and a downloadable CSV.")
        files = gr.File(file_count="multiple", type="filepath", label="Fundus images")
        run_btn = gr.Button("Score all", variant="primary")
        table = gr.Dataframe(headers=["file", "status", "stage", "grade", "referable_prob", "action"],
                             label="Triage list (most severe first)")
        csv_out = gr.File(label="Download CSV")

    assess_btn.click(grade, inp, [result_md, proc_img, cam_img, ctx_state, chatbot])
    msg.submit(chat_fn, [msg, chatbot, ctx_state], [chatbot, msg])
    send.click(chat_fn, [msg, chatbot, ctx_state], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)
    run_btn.click(batch, files, [table, csv_out])


if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("GRADIO_SHARE")), theme=gr.themes.Soft())
