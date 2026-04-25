"""
teacher.py
TeacherPanel - three independent LLM calls on the same question, each with a
different prompting style (concise / step-by-step / devil's advocate).
Run in parallel with asyncio.gather().
"""

from __future__ import annotations

import asyncio
import re
from typing import List

import httpx


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


_STYLE_PROMPTS = {
    "concise": (
        "You are a concise expert. Answer the question directly in 2-3 sentences max. "
        "No padding, no preamble. End your response with exactly one line of the form: "
        "CONFIDENCE: X.X (where X.X is a float between 0.0 and 1.0)."
    ),
    "step_by_step": (
        "You are a careful reasoner. Walk through the problem step by step. Show your "
        "working. Then state your final answer. End your response with exactly one line "
        "of the form: CONFIDENCE: X.X (where X.X is a float between 0.0 and 1.0)."
    ),
    "devils_advocate": (
        "You are a devil's advocate. First argue for the strongest opposing or "
        "counterintuitive answer (1-2 sentences). Then state what you actually believe "
        "the correct answer is and briefly justify it. End your response with exactly "
        "one line of the form: CONFIDENCE: X.X (where X.X is a float between 0.0 and 1.0)."
    ),
}


def _parse_confidence(text: str) -> float:
    if not text:
        return 0.5
    match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not match:
        return 0.5
    try:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 10.0
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5


def _strip_confidence_line(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = [l for l in lines if not re.match(r"^\s*CONFIDENCE\s*:", l, re.IGNORECASE)]
    return "\n".join(cleaned).strip()


class TeacherPanel:
    def __init__(self, groq_api_key: str, model: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key
        self.model = model

    async def _call_teacher(self, question: str, style: str) -> dict:
        system_prompt = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["concise"])
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        raw = ""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"[Teacher:{style}] Groq error: {e}")

        confidence = _parse_confidence(raw)
        body = _strip_confidence_line(raw)
        return {
            "style": style,
            "answer": body,
            "reasoning": body,
            "confidence": confidence,
            "raw": raw,
        }

    async def answer_all(self, question: str) -> List[dict]:
        tasks = [
            self._call_teacher(question, "concise"),
            self._call_teacher(question, "step_by_step"),
            self._call_teacher(question, "devils_advocate"),
        ]
        return await asyncio.gather(*tasks)
