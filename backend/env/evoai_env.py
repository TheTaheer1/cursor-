"""
evoai_env.py
OpenEnv-compliant environment wrapper for EvoAI Lab.
Exposes the training loop as reset() / step() / state() interface.
"""

from __future__ import annotations

import os
from typing import Any, Optional

# OpenEnv import with safe fallback. The OpenEnv project still ships under a
# few naming conventions across releases; we try the most common ones and fall
# back to a permissive base class so the wrapper still works in environments
# where openenv is not yet installed.
try:
    from openenv import Environment as _OpenEnvBase  # type: ignore
except Exception:
    try:
        from openenv.env import Environment as _OpenEnvBase  # type: ignore
    except Exception:
        try:
            from openenv import MCPEnvironment as _OpenEnvBase  # type: ignore
        except Exception:
            class _OpenEnvBase:  # minimal fallback
                pass

from backend.core.pipeline import EvoAIPipeline


class EvoAIEnv(_OpenEnvBase):
    """OpenEnv wrapper around the EvoAI Lab self-improving training loop."""

    metadata = {"name": "evoai-lab", "theme": "self-improvement"}

    def __init__(self, config: Optional[dict] = None):
        try:
            super().__init__()  # base class may or may not require init
        except Exception:
            pass
        self.config = config or {}
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.pipeline = EvoAIPipeline(groq_key, self.config)
        self._state: dict = {
            "step": 0,
            "calibration_map": self.pipeline.calibration_map.snapshot(),
            "reward_history": [],
            "last_step_result": {},
        }

    def reset(self) -> dict:
        self.pipeline.reset()
        self._state = {
            "step": 0,
            "calibration_map": self.pipeline.calibration_map.snapshot(),
            "reward_history": [],
            "last_step_result": {},
        }
        return self._state

    async def step(self, action: Optional[dict] = None) -> dict:
        result = await self.pipeline.run_step()
        self._state["step"] = self.pipeline.step
        self._state["calibration_map"] = self.pipeline.calibration_map.snapshot()
        self._state["reward_history"] = list(self.pipeline.reward_history)
        self._state["last_step_result"] = result

        snapshot = self._state["calibration_map"]
        observation = {
            "step": self._state["step"],
            "calibration_map": snapshot,
            "zone_c_count": snapshot.get("zone_c_count", 0),
            "zone_b_count": snapshot.get("zone_b_count", 0),
            "green_count": snapshot.get("green_count", 0),
            "last_probe": result.get("probe", {}),
            "last_question": result.get("question", {}),
        }
        return {
            "observation": observation,
            "reward": result.get("reward", 0.0),
            "done": False,
            "info": result,
        }

    def state(self) -> dict:
        snapshot = self.pipeline.calibration_map.snapshot()
        return {
            "step": self.pipeline.step,
            "calibration_map": snapshot,
            "zone_c_count": snapshot.get("zone_c_count", 0),
            "zone_b_count": snapshot.get("zone_b_count", 0),
            "green_count": snapshot.get("green_count", 0),
            "reward_history": list(self.pipeline.reward_history),
            "filter_pass_rate": self.pipeline.disagreement_detector.pass_rate,
            "filter_pass_rate_history": list(self.pipeline.filter_pass_rate_history),
            "last_step_result": self.pipeline.last_step_result,
        }

    def close(self) -> None:
        try:
            self.pipeline.dataset_builder.flush_to_disk()
        except Exception as e:
            print(f"[EvoAIEnv] close flush error: {e}")
