"""
critic.py
Evaluates the reasoning quality of each teacher's answer (not just whether the
final answer is correct). Returns per-axis scores: logical, completeness, no_shortcuts.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

import httpx


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


_CRITIC_SYSTEM = (
    "You are a reasoning quality evaluator. You will receive a question and an answer "
    "with reasoning. Score the reasoning on three axes:\n"
    "  (1) Logical soundness 0-10: are the steps valid?\n"
    "  (2) Completeness 0-10: are steps missing?\n"
    "  (3) No shortcuts 0-10: does it avoid circular reasoning?\n"
    "Return strict JSON of the form: "
    "{\"logical\": int, \"completeness\": int, \"no_shortcuts\": int, "
    "\"flags\": [\"short string flags\"], \"overall\": float}.\n"
    "No prose, no code fences."
)


def _safe_parse_json(raw: str) -> dict:
    if not raw:
        return {}
    try:
        cleaned = raw.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(cleaned[start : end + 1])
    except Exception:
        return {}


class Critic:
    def __init__(self, groq_api_key: str, model: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key
        self.model = model

    async def _call(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            print(f"[Critic] Groq error: {e}")
            return ""

    async def _evaluate_one(self, question: str, teacher_output: dict) -> dict:
        user = (
            f"Question:\n{question}\n\n"
            f"Style: {teacher_output.get('style')}\n"
            f"Answer/Reasoning:\n{teacher_output.get('answer','')}\n\n"
            "Score now."
        )
        raw = await self._call(_CRITIC_SYSTEM, user)
        parsed = _safe_parse_json(raw)
        logical = int(parsed.get("logical", 5) or 5)
        completeness = int(parsed.get("completeness", 5) or 5)
        no_shortcuts = int(parsed.get("no_shortcuts", 5) or 5)
        flags = parsed.get("flags", []) or []
        overall = parsed.get("overall")
        if overall is None:
            overall = (logical + completeness + no_shortcuts) / 3.0
        try:
            overall = float(overall)
        except Exception:
            overall = (logical + completeness + no_shortcuts) / 3.0

        result = dict(teacher_output)
        result.update(
            {
                "logical": logical,
                "completeness": completeness,
                "no_shortcuts": no_shortcuts,
                "flags": flags,
                "reasoning_score": (logical + completeness + no_shortcuts) / 3.0,
                "overall": overall,
            }
        )
        return result

    async def evaluate_all(self, question: str, teacher_outputs: List[dict]) -> List[dict]:
        tasks = [self._evaluate_one(question, t) for t in teacher_outputs]
        return await asyncio.gather(*tasks)
