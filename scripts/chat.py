"""
Interactive RetinaBot session.
===================================================================

    python scripts/chat.py                     # just chat about diabetic retinopathy
    python scripts/chat.py --image path.png     # grade the image first, then chat
                                               #   with that result as context

Type a question and press Enter. Commands: 'quit' / 'exit' to leave,
'reset' to drop the image context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config


def grade_image(cfg, path: str):
    """Run the quality gate + trained model on one image; return a result_context
    dict (or None if the image is rejected / no checkpoints)."""
    import cv2
    import torch
    from src.quality_gate import QualityGate
    from src.preprocess import make_preprocessor
    from src.augment import build_eval_aug
    from src.model import DRModel, predict_stage
    from src import evaluate as E

    img = cv2.imread(path)
    if img is None:
        print(f"could not read {path}"); return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    report = QualityGate.from_config(cfg).assess(img)
    print("\n" + report.as_text())
    if not report.passed:
        return None

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = E.load_fold_models(cfg, dev)[0]
    except FileNotFoundError:
        print("(no trained checkpoints found — chatting without a result context)")
        return None

    S = cfg["image"]["size_stage2"]
    x = build_eval_aug(S, cfg["image"]["mean"], cfg["image"]["std"])(
        image=make_preprocessor(cfg)(img))["image"].unsqueeze(0).to(dev)
    with torch.no_grad():
        p = predict_stage(model(x))
    stage = int(p["ordinal_pred"])
    ctx = {"stage": stage,
           "confidence": float(p["softmax_prob"][0, stage]),
           "referable_prob": float(p["referable_prob"])}
    print(f"graded: stage {stage} ({cfg['classes'][stage]}), "
          f"confidence {ctx['confidence']:.0%}, referable p {ctx['referable_prob']:.0%}\n")
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="fundus image to grade before chatting")
    args = ap.parse_args()

    cfg = load_config()
    from src.chatbot import RetinaBot
    print("loading RetinaBot ...")
    bot = RetinaBot(kb_path=cfg["chatbot"]["kb_path"],
                    model_name=cfg["chatbot"]["encoder"],
                    min_score=cfg["chatbot"]["min_score"],
                    top_k=cfg["chatbot"]["top_k"])
    print(f"ready — {len(bot.snippets)} knowledge-base snippets, FAISS={bot._faiss}")

    ctx = grade_image(cfg, args.image) if args.image else None

    print("\nAsk about diabetic retinopathy (stages, referral, screening, risk "
          "factors, symptoms, treatment). 'quit' to exit, 'reset' to clear image "
          "context.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q.lower() in {"quit", "exit"}:
            break
        if q.lower() == "reset":
            ctx = None; print("(image context cleared)\n"); continue
        if not q:
            continue
        out = bot.answer(q, result_context=ctx)
        print("\nRetinaBot >", out["answer"])
        if out["citations"]:
            print("  sources:", ", ".join(out["citations"]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
