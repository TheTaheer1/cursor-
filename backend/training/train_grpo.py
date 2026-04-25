"""
train_grpo.py
GRPO fine-tuning of Llama-3-8B using the EvoAI Lab dataset.
Uses HuggingFace TRL GRPOTrainer.
Positive sample:        gold_answer
Contrastive negative:   failure_answer (with correction)
Reward:                 from EvoAI dual-axis reward function
"""

from __future__ import annotations

import json
import os
import sys
from typing import List


DATASET_PATH = os.environ.get("EVOAI_DATASET", "data/training_pairs.jsonl")
OUTPUT_DIR = os.environ.get("EVOAI_OUTPUT_DIR", "./evoai-grpo-output")
MODEL_NAME = os.environ.get("EVOAI_MODEL", "meta-llama/Meta-Llama-3-8B")


def load_dataset_from_disk(path: str = DATASET_PATH):
    if not os.path.isfile(path):
        print(
            f"[train_grpo] dataset not found at {path}. "
            "Run the EvoAI pipeline (uvicorn app:app then POST /api/run-steps) "
            "to produce training pairs first."
        )
        sys.exit(1)

    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        print(f"[train_grpo] no usable training pairs in {path}")
        sys.exit(1)

    try:
        from datasets import Dataset
    except Exception as e:
        print(f"[train_grpo] datasets not installed: {e}")
        sys.exit(1)

    return Dataset.from_list(
        [
            {
                "prompt": r.get("prompt", ""),
                "chosen": r.get("chosen", ""),
                "rejected": r.get("rejected", ""),
                "reward": float(r.get("reward", 0.0)),
            }
            for r in rows
        ]
    )


def reward_function(completions, **kwargs):
    """Look up reward by prompt index. Falls back to 0.0 if no match."""
    rewards = kwargs.get("reward", [])
    if isinstance(rewards, list) and rewards:
        return [float(r) for r in rewards[: len(completions)]] + [
            0.0
        ] * max(0, len(completions) - len(rewards))
    return [0.0] * len(completions)


def main() -> None:
    dataset = load_dataset_from_disk(DATASET_PATH)
    print(f"[train_grpo] loaded {len(dataset)} training pairs")

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        print(f"[train_grpo] missing training deps: {e}")
        print(
            "Install with: pip install torch transformers trl datasets accelerate"
        )
        sys.exit(1)

    print(f"[train_grpo] loading model {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        save_steps=50,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
    )

    trainer.train()
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train_grpo] saved final model to {final_dir}")

    rewards = [float(r) for r in dataset["reward"]]
    if rewards:
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        zone_c_reductions = sum(1 for r in rewards if r > 0)
        print(
            f"[train_grpo] final stats: "
            f"reward_mean={mean:.4f} reward_std={std:.4f} "
            f"positive_steps={zone_c_reductions}/{len(rewards)}"
        )


if __name__ == "__main__":
    main()
