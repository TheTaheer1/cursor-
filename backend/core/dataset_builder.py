"""
dataset_builder.py
Stores training pairs (gold answer + failure correction) and the reward log.
Also provides a window onto the held-out eval set.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional


class DatasetBuilder:
    def __init__(self, output_dir: str = "data/", eval_path: str = "eval/held_out_eval.json"):
        self.output_dir = output_dir
        self.eval_path = eval_path
        self.training_pairs: List[dict] = []
        self.reward_log: List[dict] = []
        self.step_counter: int = 0
        self.eval_set: List[dict] = []

        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.isfile(self.eval_path):
            try:
                with open(self.eval_path, "r", encoding="utf-8") as f:
                    self.eval_set = json.load(f)
            except Exception as e:
                print(f"[DatasetBuilder] could not load eval set {self.eval_path}: {e}")
                self.eval_set = []

    def add_training_pair(self, judgment: dict, reward: dict, extra: Optional[dict] = None) -> None:
        extra = extra or {}
        pair = {
            "prompt": judgment.get("question", ""),
            "chosen": judgment.get("gold_answer", ""),
            "rejected": judgment.get("failure_answer", ""),
            "correction": judgment.get("correction", ""),
            "reward": reward.get("total", 0.0),
            "breakdown": reward.get("breakdown", {}),
            "step": self.step_counter,
            "topic": extra.get("topic"),
            "question_type": extra.get("question_type"),
            "difficulty_tier": extra.get("difficulty_tier"),
        }
        self.training_pairs.append(pair)
        self.reward_log.append(
            {
                "step": self.step_counter,
                "reward": reward.get("total", 0.0),
                "is_positive": reward.get("is_positive", False),
            }
        )
        self.step_counter += 1

    def flush_to_disk(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        pairs_path = os.path.join(self.output_dir, "training_pairs.jsonl")
        reward_path = os.path.join(self.output_dir, "reward_log.json")
        try:
            with open(pairs_path, "a", encoding="utf-8") as f:
                for p in self.training_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            self.training_pairs = []
        except Exception as e:
            print(f"[DatasetBuilder] flush pairs failed: {e}")
        try:
            with open(reward_path, "w", encoding="utf-8") as f:
                json.dump(self.reward_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DatasetBuilder] flush reward log failed: {e}")

    def get_reward_curve(self) -> List[dict]:
        return list(self.reward_log)

    def get_recent_failures(self, n: int = 10) -> List[dict]:
        # Recent training pairs with negative reward (learning moments)
        # Search both in-memory pairs and the on-disk file when in-memory is empty.
        memory_failures = [p for p in self.training_pairs if p.get("reward", 0.0) < 0]
        if len(memory_failures) >= n:
            return list(reversed(memory_failures[-n:]))

        disk_failures: List[dict] = []
        path = os.path.join(self.output_dir, "training_pairs.jsonl")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            if row.get("reward", 0.0) < 0:
                                disk_failures.append(row)
                        except Exception:
                            continue
            except Exception:
                pass

        combined = disk_failures + memory_failures
        return list(reversed(combined[-n:]))
