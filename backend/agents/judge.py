"""
judge.py
Synthesises all loop outputs into:
  (1) one gold answer (training positive)
  (2) one failure record (training contrastive negative) with a correction explanation
"""

from __future__ import annotations

from typing import List

import httpx


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class Judge:
    def __init__(self, groq_api_key: str, model: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key
        self.model = model

    async def _groq(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 300) -> str:
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
            print(f"[Judge] Groq error: {e}")
            return ""

    async def _generate_correction(self, question: str, wrong_answer: str, gold_answer: str) -> str:
        sys = (
            "You are a tutor. In 2-3 plain sentences, explain exactly why the wrong "
            "answer fails and what the correct line of reasoning is. Be concrete and brief."
        )
        usr = (
            f"Question: {question}\n\n"
            f"Wrong answer: {wrong_answer}\n\n"
            f"Correct answer: {gold_answer}\n\n"
            "Write the correction now."
        )
        return (await self._groq(sys, usr, temperature=0.2, max_tokens=200)).strip()

    async def synthesise(
        self,
        question: str,
        teacher_outputs: List[dict],
        verified_labels: List[dict],
        probe_result: dict,
        critic_scores: List[dict],
    ) -> dict:
        # Build label lookup by style
        label_by_style = {l.get("style"): l.get("label") for l in verified_labels}
        critic_by_style = {c.get("style"): c for c in critic_scores}

        # Augment each teacher output with its verifier label and critic score
        augmented = []
        for t in teacher_outputs:
            style = t.get("style")
            augmented.append(
                {
                    **t,
                    "label": label_by_style.get(style, "unverifiable"),
                    "reasoning_score": float(
                        critic_by_style.get(style, {}).get("reasoning_score", 0.0)
                    ),
                }
            )

        correct = [a for a in augmented if a["label"] == "correct"]
        incorrect = [a for a in augmented if a["label"] == "incorrect"]

        if not correct:
            return {
                "question": question,
                "gold_answer": "UNVERIFIABLE",
                "failure_answer": "",
                "failure_reason": "No verifier-correct teacher available.",
                "correction": "",
                "is_valid_pair": False,
                "hallucination_detected": False,
            }

        best_correct = max(correct, key=lambda a: a["reasoning_score"])
        gold_answer = best_correct.get("answer", "")

        if not incorrect:
            return {
                "question": question,
                "gold_answer": gold_answer,
                "failure_answer": "",
                "failure_reason": "No incorrect teacher to contrast against.",
                "correction": "",
                "is_valid_pair": False,
                "hallucination_detected": False,
            }

        # Most-confidently-wrong teacher becomes the contrastive negative
        worst_wrong = max(incorrect, key=lambda a: float(a.get("confidence", 0.0)))
        failure_answer = worst_wrong.get("answer", "")

        correction = await self._generate_correction(question, failure_answer, gold_answer)

        # Hallucination heuristic: probe was confidently wrong AND a teacher was correct.
        hallucination = (
            (not probe_result.get("is_correct", False))
            and probe_result.get("confidence", 0) >= 7
        )

        return {
            "question": question,
            "gold_answer": gold_answer,
            "failure_answer": failure_answer,
            "failure_reason": (
                f"Most confident incorrect teacher style: {worst_wrong.get('style')} "
                f"(confidence={worst_wrong.get('confidence')})."
            ),
            "correction": correction,
            "is_valid_pair": True,
            "hallucination_detected": hallucination,
        }
