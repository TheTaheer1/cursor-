"""
adversary.py
Generates questions that target the student model's weakest calibration nodes.
Zone C nodes (high confidence + wrong) are targeted first, then Zone B.
Difficulty escalates as Zone C shrinks.
"""

from __future__ import annotations

import json
import random
from typing import List

import httpx

from backend.core.calibration_map import CalibrationMap, CalibrationNode


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class Adversary:
    def __init__(self, groq_api_key: str, model: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key
        self.model = model
        self.force_escalate: bool = False

    @staticmethod
    def _difficulty_for_step(step: int) -> str:
        if step <= 100:
            return "moderate"
        if step <= 300:
            return "hard"
        return "expert"

    def _build_system_prompt(self, weak_nodes: List[CalibrationNode], difficulty: str) -> str:
        weak_block = "\n".join(
            f"- topic={n.topic}, type={n.question_type}, tier={n.difficulty_tier}, "
            f"avg_conf={n.confidence_avg:.2f}, visits={n.visit_count}, zone={n.zone}"
            for n in weak_nodes[:5]
        ) or "- (none yet, choose any topic)"

        return (
            "You are an adversarial question generator for a calibration training loop. "
            "Your job is to produce ONE question that the student model is most likely to "
            "answer confidently AND wrongly (Zone C). Edge cases, contestable phrasing, "
            "and counterintuitive framing are encouraged. Avoid trivia or trick questions; "
            "questions must be genuinely contestable.\n\n"
            f"Target difficulty tier: {difficulty}.\n\n"
            "The student's weakest calibration nodes are:\n"
            f"{weak_block}\n\n"
            "Respond with strict JSON of the form:\n"
            "{\n"
            '  "question": "<the question>",\n'
            '  "topic": "<one of: math, code, logic, factual, planning>",\n'
            '  "question_type": "<one of: math, code, factual, reasoning>",\n'
            f'  "difficulty_tier": "{difficulty}",\n'
            '  "target_node": "<topic::question_type::difficulty_tier>"\n'
            "}\n"
            "No prose, no code fences."
        )

    def _parse_response(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            cleaned = raw.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1:
                return {}
            return json.loads(cleaned[start : end + 1])
        except Exception as e:
            print(f"[Adversary] parse error: {e}")
            return {}

    def _fallback_question(self, difficulty: str) -> dict:
        topic = random.choice(["math", "logic", "code", "factual", "planning"])
        qtype = {
            "math": "math",
            "code": "code",
            "logic": "reasoning",
            "factual": "factual",
            "planning": "reasoning",
        }[topic]
        examples = {
            "math": "Compute the sum of all primes less than 50, then divide by the number of primes used.",
            "code": "Write a Python function that returns the second largest unique element of a list, or None if it does not exist. Print its result on [4, 2, 4, 2].",
            "logic": "If exactly two of the following three statements are true, which is the false one? (a) A implies B. (b) B implies C. (c) C implies A.",
            "factual": "Which year did the Treaty of Westphalia end the Thirty Years' War, and which two treaties are usually counted as part of it?",
            "planning": "You have 3 jugs of capacity 8, 5, and 3 litres. The 8L jug is full. Plan moves to leave exactly 4L in the 8L jug.",
        }
        return {
            "question": examples[topic],
            "topic": topic,
            "question_type": qtype,
            "difficulty_tier": difficulty,
            "target_node": f"{topic}::{qtype}::{difficulty}",
        }

    async def generate_question(self, calibration_map: CalibrationMap, step: int) -> dict:
        difficulty = self._difficulty_for_step(step)
        if self.force_escalate and difficulty == "moderate":
            difficulty = "hard"

        zone_c = calibration_map.get_zone_c_nodes()
        weak = zone_c if zone_c else calibration_map.get_zone_b_nodes()

        system_prompt = self._build_system_prompt(weak, difficulty)
        user_prompt = (
            "Generate one targeted, edge-case question now. Respond with strict JSON only."
        )

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 512,
        }

        raw = ""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Adversary] Groq error: {e}")

        parsed = self._parse_response(raw)
        if not parsed or "question" not in parsed:
            parsed = self._fallback_question(difficulty)

        parsed.setdefault("difficulty_tier", difficulty)
        parsed.setdefault(
            "target_node",
            f"{parsed.get('topic','math')}::{parsed.get('question_type','reasoning')}::{parsed['difficulty_tier']}",
        )
        return parsed
