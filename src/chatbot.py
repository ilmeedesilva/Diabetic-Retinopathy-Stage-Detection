"""
M7 — RetinaBot: a retrieval-grounded diabetic-retinopathy assistant.
===================================================================

Design principle: **the bot never invents medical claims.** Every sentence in an
answer is either
  (a) a statement about *this* prediction, derived from the model's own output, or
  (b) a verbatim snippet from the curated knowledge base (`kb/dr_knowledge.jsonl`),
      shown with its source label.
If no knowledge-base snippet is relevant enough to the question, the bot says so
instead of guessing. Every answer ends with a "not a diagnosis" notice.

Stack: `sentence-transformers` (MiniLM) for embeddings + FAISS for nearest-
neighbour search (falls back to NumPy cosine if FAISS is unavailable). No LLM and
no network call at answer time — it runs fully offline inside the notebook / app.
An `use_llm` hook is left for an optional phrasing-polish layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DISCLAIMER = ("\n\n_This is general information from screening guidance, not a "
               "diagnosis. An eye-care professional makes the final decision. Seek "
               "same-day care for sudden vision loss, new floaters or flashing lights._")

_STAGE_KEYWORDS = {
    0: "no diabetic retinopathy screening interval",
    1: "mild non-proliferative diabetic retinopathy microaneurysms management",
    2: "moderate non-proliferative diabetic retinopathy referable ophthalmology referral",
    3: "severe non-proliferative diabetic retinopathy prompt referral progression",
    4: "proliferative diabetic retinopathy urgent referral neovascularisation treatment",
}

_STAGE_EXPLANATION = {
    0: ("The model graded this image as **No DR (stage 0)** — no diabetic "
        "retinopathy detected. Routine re-screening still applies."),
    1: ("The model graded this image as **Mild NPDR (stage 1)** — earliest "
        "changes (microaneurysms). Usually managed by tighter diabetes control "
        "and continued screening, not referral."),
    2: ("The model graded this image as **Moderate NPDR (stage 2)** — this is "
        "referable: an ophthalmology assessment is recommended."),
    3: ("The model graded this image as **Severe NPDR (stage 3)** — a prompt "
        "ophthalmology referral is recommended because the risk of progression "
        "is high."),
    4: ("The model graded this image as **Proliferative DR (stage 4)** — this is "
        "an urgent ophthalmology referral; it is a sight-threatening stage."),
}


# --------------------------------------------------------------------------- #
class RetinaBot:
    def __init__(self, kb_path: str | Path = "kb/dr_knowledge.jsonl",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 min_score: float = 0.28, top_k: int = 3):
        self.min_score = min_score
        self.top_k = top_k
        self.snippets = [json.loads(l) for l in Path(kb_path).read_text().splitlines() if l.strip()]

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        emb = self.encoder.encode([s["text"] for s in self.snippets],
                                  normalize_embeddings=True).astype("float32")
        self.emb = emb
        try:
            import faiss
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            self._faiss = True
        except Exception:
            self.index = None
            self._faiss = False

    # -- retrieval ------------------------------------------------------- #
    def retrieve(self, query: str, k: int | None = None) -> list[tuple[dict, float]]:
        k = k or self.top_k
        q = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        if self._faiss:
            scores, idx = self.index.search(q, k)
            pairs = [(self.snippets[i], float(s)) for i, s in zip(idx[0], scores[0])]
        else:
            sims = (self.emb @ q[0])
            order = np.argsort(-sims)[:k]
            pairs = [(self.snippets[i], float(sims[i])) for i in order]
        return [(s, sc) for s, sc in pairs if sc >= self.min_score]

    # -- result explanation (model-derived, not retrieved) ------------- #
    @staticmethod
    def explain_result(stage: int, referable_prob: float | None = None,
                       confidence: float | None = None) -> str:
        txt = _STAGE_EXPLANATION.get(int(stage), "")
        if confidence is not None:
            txt += f" Model confidence for this grade: {confidence:.0%}."
        if referable_prob is not None:
            verdict = "refer" if referable_prob >= 0.5 else "routine re-screen"
            txt += f" Referable-DR probability: {referable_prob:.0%} → suggested action: **{verdict}**."
        return txt

    # -- answer -------------------------------------------------------- #
    def answer(self, query: str, result_context: dict | None = None,
               use_llm: bool = False) -> dict[str, Any]:
        """result_context (optional): {'stage': int, 'referable_prob': float,
        'confidence': float} to prepend a prediction-specific explanation."""
        parts: list[str] = []
        citations: list[str] = []

        if result_context and "stage" in result_context:
            # Explain-the-result mode: stage explanation + guidance for that grade.
            # Biasing the query with the grade keywords keeps retrieval on-topic even
            # for very vague questions ("what now?"); specific on-topic questions
            # still surface their own snippets because the query terms also count.
            k = int(result_context["stage"])
            parts.append(self.explain_result(k, result_context.get("referable_prob"),
                                             result_context.get("confidence")))
            hits = self.retrieve(f"{query} {_STAGE_KEYWORDS.get(k, '')}")
            parts.append("Guidance relevant to this result:")
            for s, sc in hits:
                parts.append(f"• {s['text']} _[{s['source']}]_")
                citations.append(s["source"])
        else:
            # Q&A mode: answer from the knowledge base, or decline.
            hits = self.retrieve(query)
            if hits:
                parts.append("From diabetic-retinopathy screening guidance:")
                for s, sc in hits:
                    parts.append(f"• {s['text']} _[{s['source']}]_")
                    citations.append(s["source"])
            else:
                parts.append("I don't have guideline-grounded information on that. "
                             "I can help with diabetic retinopathy stages, referral, "
                             "risk factors, screening, symptoms and treatment — for "
                             "anything else, please ask an eye-care professional.")

        answer = "\n\n".join(parts) + _DISCLAIMER

        if use_llm:  # optional phrasing polish — left as an integration hook
            answer = _llm_polish(answer)

        return {"answer": answer,
                "citations": sorted(set(citations)),
                "retrieved": [{"id": s["id"], "score": round(sc, 3), "source": s["source"]}
                              for s, sc in hits]}


def _llm_polish(text: str) -> str:  # pragma: no cover - disabled by default
    """Placeholder: an instruction-tuned LLM could rephrase `text` into one flowing
    paragraph while being told to change no facts and keep every _[source]_ tag."""
    return text


# --------------------------------------------------------------------------- #
# Smoke test: `python -m src.chatbot`                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    bot = RetinaBot()
    print(f"loaded {len(bot.snippets)} snippets | FAISS: {bot._faiss}\n")
    for q in ["what does moderate NPDR mean and do I need to see a doctor?",
              "how often should someone with diabetes get their eyes checked?",
              "can I still drive?"]:
        print("Q:", q)
        out = bot.answer(q, result_context={"stage": 2, "referable_prob": 0.71, "confidence": 0.63})
        print(out["answer"])
        print("citations:", out["citations"], "\n" + "-" * 70)
