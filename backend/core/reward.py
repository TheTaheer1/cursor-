"""
reward.py
Dual-axis reward function for EvoAI Lab.

Two axes:
  1. Accuracy  - was the answer correct?
  2. Calibration - was the model's confidence aligned with its actual correctness?

The most heavily penalised case is the confident hallucination (Zone C),
which combines wrong_high_confidence (-0.40) with hallucination (-0.50).
"""

from __future__ import annotations

from typing import List, Optional


class RewardCalculator:
    """Computes the dual-axis reward and breakdown for a single training step."""

    CORRECT_ANSWER = 0.35
    ZONE_C_TO_B_SHIFT = 0.30
    HIGH_REASONING = 0.20
    APPROPRIATE_UNCERTAINTY = 0.15
    WRONG_HIGH_CONFIDENCE = -0.40
    CORRECT_LOW_CONFIDENCE = -0.15
    OVER_REFUSAL = -0.20
    HALLUCINATION = -0.50

    def __init__(self):
        pass

    def compute(
        self,
        probe_result: dict,
        judgment: dict,
        critic_scores: List[dict],
        previous_zone: Optional[str] = None,
        question_marked_answerable: bool = True,
    ) -> dict:
        total = 0.0
        breakdown: dict = {}

        is_correct = bool(probe_result.get("is_correct", False))
        confidence = int(probe_result.get("confidence", 0))
        current_zone = probe_result.get("zone", "zone_b")

        # --- Accuracy axis ---
        if is_correct:
            total += self.CORRECT_ANSWER
            breakdown["correct_answer"] = self.CORRECT_ANSWER

        if (not is_correct) and confidence >= 7:
            total += self.WRONG_HIGH_CONFIDENCE
            breakdown["wrong_high_confidence"] = self.WRONG_HIGH_CONFIDENCE

        if is_correct and confidence < 4:
            total += self.CORRECT_LOW_CONFIDENCE
            breakdown["correct_low_confidence"] = self.CORRECT_LOW_CONFIDENCE

        # --- Calibration axis ---
        if previous_zone == "zone_c" and current_zone in ("zone_b", "green"):
            total += self.ZONE_C_TO_B_SHIFT
            breakdown["zone_shift"] = self.ZONE_C_TO_B_SHIFT

        # If model was confidently correct, certainty was justified - no bonus, no penalty.
        if is_correct and confidence >= 7:
            breakdown["justified_certainty"] = 0.0

        # If model was wrong but appropriately uncertain - reward partial calibration.
        if (not is_correct) and confidence < 5:
            total += self.APPROPRIATE_UNCERTAINTY
            breakdown["appropriate_uncertainty"] = self.APPROPRIATE_UNCERTAINTY

        # --- Reasoning axis ---
        if critic_scores:
            valid_scores = [
                float(c.get("reasoning_score", 0.0))
                for c in critic_scores
                if isinstance(c, dict)
            ]
            if valid_scores:
                avg_reasoning = sum(valid_scores) / len(valid_scores)
                if avg_reasoning > 8.0:
                    total += self.HIGH_REASONING
                    breakdown["high_reasoning"] = self.HIGH_REASONING

        # --- Hallucination ---
        if judgment.get("hallucination_detected", False):
            total += self.HALLUCINATION
            breakdown["hallucination"] = self.HALLUCINATION

        # --- Over-refusal ---
        student_answer = (probe_result.get("student_answer") or "").strip().lower()
        refusal_prefixes = ("i cannot", "i'm unable", "i am unable", "i don't know")
        if (
            student_answer.startswith(refusal_prefixes)
            and judgment.get("is_valid_pair", False)
            and question_marked_answerable
        ):
            total += self.OVER_REFUSAL
            breakdown["over_refusal"] = self.OVER_REFUSAL

        total = round(total, 4)
        return {
            "total": total,
            "breakdown": breakdown,
            "is_positive": total > 0,
        }
