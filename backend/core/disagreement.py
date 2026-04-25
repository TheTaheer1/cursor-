"""
disagreement.py
Computes pairwise semantic similarity between teacher answers.
Filters out questions where teachers agree (those are too easy and produce no signal).
"""

from __future__ import annotations

from typing import List

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - allow import without the dep installed
    SentenceTransformer = None  # type: ignore
    import math
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


def _cosine(a, b) -> float:
    if _SENTENCE_TRANSFORMERS_AVAILABLE:
        denom = (float((a * a).sum()) ** 0.5) * (float((b * b).sum()) ** 0.5)
        if denom == 0:
            return 0.0
        return float((a * b).sum()) / denom
    # Fallback: lexical Jaccard similarity
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class DisagreementDetector:
    """Filters teacher outputs - returns True only when teachers meaningfully disagree."""

    def __init__(self, threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2"):
        self.threshold = float(threshold)
        self.total_seen = 0
        self.total_passed = 0
        self.model_name = model_name
        self._model = None
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as e:  # pragma: no cover
                print(f"[DisagreementDetector] could not load {model_name}: {e}")
                self._model = None

    def _embed(self, texts: List[str]):
        if self._model is None:
            return texts  # use raw strings; _cosine falls back to Jaccard
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def filter(self, teacher_outputs: List[dict]) -> bool:
        self.total_seen += 1

        answers = [str(t.get("answer", "")) for t in teacher_outputs]
        if len(answers) < 2:
            return False

        embeddings = self._embed(answers)

        sims: List[float] = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                sims.append(_cosine(embeddings[i], embeddings[j]))

        avg_sim = sum(sims) / len(sims) if sims else 1.0

        if avg_sim > self.threshold:
            return False

        self.total_passed += 1
        return True

    @property
    def pass_rate(self) -> float:
        return self.total_passed / max(self.total_seen, 1)
