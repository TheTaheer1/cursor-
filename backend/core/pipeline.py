"""
pipeline.py
Master orchestrator for the EvoAI Lab training loop.
Runs indefinitely: adversary -> teachers -> disagreement filter -> verifier ->
calibration probe -> critic -> judge -> reward -> dataset builder -> loop.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from backend.agents.adversary import Adversary
from backend.agents.teacher import TeacherPanel
from backend.agents.calibration_probe import CalibrationProbe
from backend.agents.critic import Critic
from backend.agents.judge import Judge
from backend.core.calibration_map import CalibrationMap
from backend.core.dataset_builder import DatasetBuilder
from backend.core.disagreement import DisagreementDetector
from backend.core.reward import RewardCalculator
from backend.core.verifier import Verifier


class EvoAIPipeline:
    def __init__(self, groq_api_key: str, config: Optional[dict] = None):
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.groq_api_key = groq_api_key
        self.config = config or {}

        student_model = self.config.get("student_model", "llama3-8b-8192")
        teacher_model = self.config.get("teacher_model", "llama3-8b-8192")

        self.calibration_map = CalibrationMap()
        self.adversary = Adversary(groq_api_key, model=teacher_model)
        self.teacher_panel = TeacherPanel(groq_api_key, model=teacher_model)
        self.disagreement_detector = DisagreementDetector(
            threshold=self.config.get("disagreement_threshold", 0.85)
        )
        self.verifier = Verifier(
            groq_api_key,
            faiss_index_path=self.config.get("faiss_index_path", "eval/faiss_index"),
            model=teacher_model,
        )
        self.calibration_probe = CalibrationProbe(
            groq_api_key,
            calibration_map=self.calibration_map,
            student_model=student_model,
        )
        self.critic = Critic(groq_api_key, model=teacher_model)
        self.judge = Judge(groq_api_key, model=teacher_model)

        self.dataset_builder = DatasetBuilder(
            output_dir=self.config.get("output_dir", "data/"),
            eval_path=self.config.get("eval_path", "eval/held_out_eval.json"),
        )
        self.reward_calculator = RewardCalculator()

        self.step = 0
        self.reward_history: List[dict] = []
        self.filter_pass_rate_history: List[float] = []
        self.last_step_result: dict = {}

    def reset(self) -> None:
        self.calibration_map = CalibrationMap()
        self.calibration_probe.calibration_map = self.calibration_map
        self.disagreement_detector = DisagreementDetector(
            threshold=self.config.get("disagreement_threshold", 0.85)
        )
        self.dataset_builder = DatasetBuilder(
            output_dir=self.config.get("output_dir", "data/"),
            eval_path=self.config.get("eval_path", "eval/held_out_eval.json"),
        )
        self.step = 0
        self.reward_history = []
        self.filter_pass_rate_history = []
        self.last_step_result = {}

    async def run_step(self) -> dict:
        step_idx = self.step

        # Step 1 - adversary picks a question
        question_data = await self.adversary.generate_question(self.calibration_map, self.step)

        # Step 2 - three teachers answer in parallel
        teacher_outputs = await self.teacher_panel.answer_all(question_data["question"])

        # Step 3 - disagreement filter: skip when teachers all agree
        passed = self.disagreement_detector.filter(teacher_outputs)
        if not passed:
            self.step += 1
            self.last_step_result = {
                "step": step_idx,
                "skipped": True,
                "reason": "teachers_agree",
                "question": question_data,
                "filter_pass_rate": self.disagreement_detector.pass_rate,
                "calibration_map": self.calibration_map.snapshot(),
                "reward": 0.0,
            }
            return self.last_step_result

        # Step 4 - verifier provides ground truth
        verified = await self.verifier.verify_all(question_data, teacher_outputs)
        gold_label = verified.get("gold_label", "UNVERIFIABLE")

        # Step 5 - probe the student model
        probe_result = await self.calibration_probe.probe(
            question_data["question"],
            gold_label,
            question_data.get("topic", "logic"),
            question_data.get("question_type", "reasoning"),
            question_data.get("difficulty_tier", "moderate"),
        )

        # Step 6a - critic scores reasoning
        critic_scores = await self.critic.evaluate_all(question_data["question"], teacher_outputs)

        # Step 6b - judge synthesises gold + failure pair
        judgment = await self.judge.synthesise(
            question_data["question"],
            teacher_outputs,
            verified.get("labels", []),
            probe_result,
            critic_scores,
        )

        # Step 7 - reward
        previous_zone = probe_result.get("previous_zone")
        reward = self.reward_calculator.compute(
            probe_result,
            judgment,
            critic_scores,
            previous_zone=previous_zone,
            question_marked_answerable=True,
        )

        # Step 8 - dataset record (only valid pairs)
        if judgment.get("is_valid_pair"):
            self.dataset_builder.add_training_pair(
                judgment,
                reward,
                extra={
                    "topic": question_data.get("topic"),
                    "question_type": question_data.get("question_type"),
                    "difficulty_tier": question_data.get("difficulty_tier"),
                },
            )

        self.reward_history.append({"step": step_idx, "reward": reward["total"]})

        # Track filter pass rate every 10 steps
        if (self.step + 1) % 10 == 0:
            rate = self.disagreement_detector.pass_rate
            self.filter_pass_rate_history.append(rate)
            if rate < 0.30:
                print(f"[Pipeline] WARN filter_pass_rate={rate:.2f} below 0.30 - escalating adversary")
                self.adversary.force_escalate = True

        self.step += 1

        result = {
            "step": step_idx,
            "skipped": False,
            "question": question_data,
            "teacher_outputs": teacher_outputs,
            "verified": verified,
            "probe": probe_result,
            "critic_scores": critic_scores,
            "judgment": judgment,
            "reward": reward["total"],
            "reward_breakdown": reward["breakdown"],
            "calibration_map": self.calibration_map.snapshot(),
            "filter_pass_rate": self.disagreement_detector.pass_rate,
            "failure": (
                {
                    "question": question_data["question"],
                    "topic": question_data.get("topic"),
                    "question_type": question_data.get("question_type"),
                    "difficulty_tier": question_data.get("difficulty_tier"),
                    "failure_answer": judgment.get("failure_answer", ""),
                    "correction": judgment.get("correction", ""),
                }
                if judgment.get("is_valid_pair") and reward["total"] < 0
                else None
            ),
        }
        self.last_step_result = result
        return result

    async def run_loop(self, max_steps: int = 1000) -> List[dict]:
        results: List[dict] = []
        for i in range(max_steps):
            try:
                r = await self.run_step()
                results.append(r)
                print(
                    f"[Pipeline] step={r.get('step')} reward={r.get('reward'):.3f} "
                    f"skipped={r.get('skipped')} pass_rate={r.get('filter_pass_rate', 0):.2f}"
                )
            except Exception as e:
                print(f"[Pipeline] step error: {e}")
            if (i + 1) % 50 == 0:
                self.dataset_builder.flush_to_disk()
        self.dataset_builder.flush_to_disk()
        return results
