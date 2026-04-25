"""
run_ece.py
Computes Expected Calibration Error (ECE) on the held-out eval set.
ECE measures how well confidence scores match actual accuracy.
A perfectly calibrated model has ECE = 0.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List


def compute_ece(predictions: List[dict], n_bins: int = 10) -> float:
    if not predictions:
        return 0.0

    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = len(predictions)
    ece = 0.0

    for lo, hi in bins:
        bucket = [
            p for p in predictions
            if (lo <= p["confidence"] < hi) or (hi == 1.0 and p["confidence"] == 1.0)
        ]
        if not bucket:
            continue
        avg_conf = sum(p["confidence"] for p in bucket) / len(bucket)
        avg_acc = sum(1.0 for p in bucket if p["is_correct"]) / len(bucket)
        weight = len(bucket) / total
        ece += weight * abs(avg_conf - avg_acc)

    return ece


async def run_eval(env, eval_path: str = "eval/held_out_eval.json") -> dict:
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"eval set not found: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    predictions: List[dict] = []
    for q in eval_set:
        try:
            probe_result = await env.pipeline.calibration_probe.probe(
                question=q["question"],
                correct_answer=q.get("gold_answer", ""),
                topic=q.get("topic", "factual"),
                question_type=q.get("question_type", "factual"),
                difficulty_tier=q.get("difficulty_tier", "moderate"),
            )
            predictions.append(
                {
                    "id": q.get("id"),
                    "confidence": probe_result["confidence"] / 10.0,
                    "is_correct": bool(probe_result["is_correct"]),
                    "zone": probe_result["zone"],
                }
            )
        except Exception as e:
            print(f"[run_ece] error on {q.get('id')}: {e}")

    if not predictions:
        return {"ece": 0.0, "accuracy": 0.0, "zone_c_fraction": 0.0, "n_questions": 0}

    ece = compute_ece(predictions, n_bins=10)
    accuracy = sum(1 for p in predictions if p["is_correct"]) / len(predictions)
    zone_c_fraction = sum(1 for p in predictions if p["zone"] == "zone_c") / len(predictions)

    return {
        "ece": round(ece, 4),
        "accuracy": round(accuracy, 4),
        "zone_c_fraction": round(zone_c_fraction, 4),
        "n_questions": len(predictions),
    }


async def _main_async() -> None:
    from backend.env.evoai_env import EvoAIEnv

    env = EvoAIEnv()
    result = await run_eval(env)
    print("=== EvoAI Lab held-out evaluation ===")
    print(json.dumps(result, indent=2))


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
