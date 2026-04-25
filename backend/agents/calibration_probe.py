"""
calibration_probe.py
Queries the student model for its answer + confidence on a question,
then plots the (confidence, correctness) pair on the calibration map.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

from backend.core.calibration_map import CalibrationMap, ZONE_C, ZONE_B, ZONE_GREEN


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


_PROBE_SYSTEM = (
    "Answer the question briefly and clearly. Then end your response with EXACTLY one "
    "final line of the form:\nCONFIDENCE: X\n"
    "where X is an integer 0-10. 0 = completely unsure, 10 = completely certain."
)


def _parse_confidence_int(text: str) -> int:
    if not text:
        return 5
    match = re.search(r"CONFIDENCE\s*:\s*(\d+)", text, re.IGNORECASE)
    if not match:
        return 5
    try:
        val = int(match.group(1))
        return max(0, min(10, val))
    except Exception:
        return 5


def _strip_confidence_line(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = [l for l in lines if not re.match(r"^\s*CONFIDENCE\s*:", l, re.IGNORECASE)]
    return "\n".join(cleaned).strip()


class CalibrationProbe:
    def __init__(
        self,
        groq_api_key: str,
        calibration_map: Optional[CalibrationMap] = None,
        student_model: str = "llama3-8b-8192",
    ):
        self.groq_api_key = groq_api_key
        self.student_model = student_model
        self.calibration_map = calibration_map

    async def _groq(self, system: str, user: str, temperature: float = 0.5, max_tokens: int = 400) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.student_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"[CalibrationProbe] Groq error: {e}")
            return ""

    async def _semantic_match(self, student_answer: str, correct_answer: str, question: str) -> bool:
        if not student_answer or not correct_answer:
            return False
        if correct_answer.strip().upper() == "UNVERIFIABLE":
            return False
        # Quick lexical shortcut for short numeric / exact answers
        sa = student_answer.strip().lower()
        ca = correct_answer.strip().lower()
        if ca and ca in sa:
            return True

        sys = (
            "You are a strict semantic grader. Decide whether the student's answer is "
            "equivalent to the reference. Reply with exactly one word: YES or NO."
        )
        usr = (
            f"Question: {question}\n\n"
            f"Reference answer: {correct_answer}\n\n"
            f"Student answer: {student_answer}\n\n"
            "Are they equivalent? Reply YES or NO."
        )
        out = await self._groq(sys, usr, temperature=0.0, max_tokens=4)
        return out.strip().upper().startswith("YES")

    @staticmethod
    def _zone_for(confidence: int, is_correct: bool) -> str:
        if is_correct:
            return ZONE_GREEN
        if confidence >= 7:
            return ZONE_C
        return ZONE_B

    async def probe(
        self,
        question: str,
        correct_answer: str,
        topic: str,
        question_type: str,
        difficulty_tier: str,
    ) -> dict:
        raw = await self._groq(_PROBE_SYSTEM, question, temperature=0.5, max_tokens=400)
        confidence = _parse_confidence_int(raw)
        student_answer = _strip_confidence_line(raw)

        is_correct = await self._semantic_match(student_answer, correct_answer, question)
        zone = self._zone_for(confidence, is_correct)

        previous_zone = None
        if self.calibration_map is not None:
            key = f"{topic}::{question_type}::{difficulty_tier}"
            existing = self.calibration_map.nodes.get(key)
            if existing:
                previous_zone = existing.zone
            self.calibration_map.update_node(
                topic=topic,
                question_type=question_type,
                difficulty_tier=difficulty_tier,
                zone=zone,
                confidence=confidence / 10.0,
                is_correct=is_correct,
            )

        return {
            "student_answer": student_answer,
            "confidence": confidence,
            "is_correct": is_correct,
            "zone": zone,
            "previous_zone": previous_zone,
            "node_key": f"{topic}::{question_type}::{difficulty_tier}",
        }
