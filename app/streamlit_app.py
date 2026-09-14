"""
M8 (alternate UI) — DR-Triage prototype, Streamlit version.
===================================================================

Same pipeline as app/app.py (quality gate -> grade -> Grad-CAM++ -> RetinaBot),
rebuilt in Streamlit so it can be hosted for free on Streamlit Community Cloud,
which — unlike Hugging Face Spaces / Render's compute tiers — does not require
card verification for its free tier.

Run locally: streamlit run app/streamlit_app.py
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
import torch.nn.functional as F
import streamlit as st

from src.data import load_config
from src.quality_gate import QualityGate
from src.preprocess import make_preprocessor
from src.augment import build_eval_aug
from src import evaluate as E
from src.chatbot import RetinaBot

st.set_page_config(page_title="DR-Triage", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model + assistant …")
def load_everything():
    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gate = QualityGate.from_config(cfg)
    pre = make_preprocessor(cfg)
    eval_tf = build_eval_aug(cfg["image"]["size_stage2"], cfg["image"]["mean"], cfg["image"]["std"])
    try:
        models = E.load_fold_models(cfg, dev)
        cap = os.environ.get("DR_TRIAGE_MAX_MODELS")  # cap ensemble size on a low-RAM host
        if cap and int(cap) > 0:
            models = models[: int(cap)]
    except FileNotFoundError:
        models = []
    bot = RetinaBot(kb_path=cfg["chatbot"]["kb_path"], model_name=cfg["chatbot"]["encoder"],
                    min_score=cfg["chatbot"]["min_score"], top_k=cfg["chatbot"]["top_k"])
    return cfg, dev, gate, pre, eval_tf, models, bot


CFG, DEV, GATE, PRE, EVAL_TF, MODELS, BOT = load_everything()
CLASSES = CFG["classes"]
T = float(CFG.get("inference", {}).get("temperature", 1.0))
REF_THR = float(CFG.get("inference", {}).get("referable_threshold", 0.5))


def predict(img_rgb: np.ndarray):
    x = EVAL_TF(image=PRE(img_rgb))["image"].unsqueeze(0).to(DEV)
    logit = ordp = refp = 0.0
    with torch.no_grad():
        for m in MODELS:
            o = m(x)
            logit = logit + o["stage"].float()
            ordp = ordp + o["ordinal"].sigmoid().float()
            refp = refp + o["referable"].sigmoid().float()
    n = max(len(MODELS), 1)
    stage_probs = F.softmax(logit / n / T, dim=1)[0].cpu().numpy()
    ord_probs = (ordp / n)[0].cpu().numpy()
    stage = int((ord_probs > 0.5).sum())
    return stage, stage_probs, float((refp / n)[0])


def overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_r = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 255).astype(np.uint8)


def read_upload(file) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
st.title("🩺 DR-Triage — Diabetic Retinopathy Screening Assistant")
st.caption("Explainable, order-aware DR staging with a referral decision, an "
          "image-quality gate and a guideline-grounded assistant. "
          "**Screening aid — not a diagnosis.**")

if "ctx" not in st.session_state:
    st.session_state.ctx = None
if "chat" not in st.session_state:
    st.session_state.chat = []

tab1, tab2, tab3 = st.tabs(["Grade an image", "Ask RetinaBot", "Batch triage"])

with tab1:
    up = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg"])
    if up is not None and st.button("Assess", type="primary"):
        img = read_upload(up)
        rep = GATE.assess(img)
        if not rep.passed:
            st.error("**Image rejected before grading**\n\n" +
                    "\n".join(f"- {r}" for r in rep.reasons))
        elif not MODELS:
            st.warning("Quality OK, but no trained checkpoints were found on this host.")
        else:
            proc = PRE(img)
            stage, sp, refp = predict(img)
            conf = float(sp[stage])
            refer = refp >= REF_THR
            cam = E.gradcam_plus_plus(MODELS[0], EVAL_TF(image=proc)["image"].unsqueeze(0),
                                      [stage], DEV)[0]
            ov = overlay(proc, cam)

            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Original", use_container_width=True)
            c2.image(proc, caption="Preprocessed", use_container_width=True)
            c3.image(ov, caption="Grad-CAM++ — where the model looked", use_container_width=True)

            st.markdown(f"## Grade {stage} — {CLASSES[stage]}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Calibrated confidence", f"{conf:.0%}")
            m2.metric("Referable-DR probability", f"{refp:.0%}")
            m3.metric("Recommended action", "REFER" if refer else "Re-screen")
            st.bar_chart(pd.Series(sp, index=[CLASSES[i] for i in range(len(CLASSES))]))

            ctx = {"stage": stage, "confidence": conf, "referable_prob": refp}
            st.session_state.ctx = ctx
            intro = BOT.answer("What does this result mean for me?", result_context=ctx)["answer"]
            st.session_state.chat = [("assistant", intro)]
            st.info("Explanation added to the **Ask RetinaBot** tab →")

with tab2:
    st.caption("Grade an image first and the assistant starts with an explanation of "
              "that result. Answers are drawn only from cited screening-guidance snippets.")
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Ask about your result or diabetic retinopathy…")
    if q:
        st.session_state.chat.append(("user", q))
        out = BOT.answer(q, result_context=st.session_state.ctx)
        st.session_state.chat.append(("assistant", out["answer"]))
        st.rerun()

with tab3:
    st.caption("Upload multiple fundus images; get a severity-sorted triage list "
              "and a downloadable CSV.")
    files = st.file_uploader("Fundus images", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True)
    if files and st.button("Score all", type="primary"):
        rows = []
        for f in files:
            img = read_upload(f)
            rep = GATE.assess(img)
            if not rep.passed:
                rows.append([f.name, "rejected", "", rep.reasons[0][:48], "", "retake image"])
                continue
            if not MODELS:
                rows.append([f.name, "no model", "", "", "", ""])
                continue
            stage, sp, refp = predict(img)
            rows.append([f.name, "graded", stage, CLASSES[stage], round(refp, 2),
                        "REFER" if refp >= REF_THR else "re-screen"])
        df = pd.DataFrame(rows, columns=["file", "status", "stage", "grade",
                                         "referable_prob", "action"])
        df["_s"] = pd.to_numeric(df["stage"], errors="coerce").fillna(-1)
        df = df.sort_values(["_s", "referable_prob"], ascending=False).drop(columns="_s").reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "dr_triage_batch.csv")
