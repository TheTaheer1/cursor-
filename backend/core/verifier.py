"""
verifier.py
Ground-truth verification for teacher answers.
Uses real execution (subprocess for code, sympy/eval for math) instead of
LLM opinion wherever possible. Falls back to Groq for factual / reasoning.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from typing import List

import httpx


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _extract_number(text: str):
    if text is None:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", str(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _extract_code_block(text: str) -> str:
    if not text:
        return ""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


class Verifier:
    """Routes verification to the correct backend based on question_type."""

    def __init__(self, groq_api_key: str, faiss_index_path: str = "eval/faiss_index", model: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key
        self.model = model
        self.faiss_index_path = faiss_index_path
        self.faiss_available = os.path.isdir(faiss_index_path) or os.path.isfile(faiss_index_path)
        self._faiss_index = None
        if not self.faiss_available:
            print("[Verifier] FAISS index not found - falling back to LLM-based factual verification")

    async def _groq_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Verifier] Groq error: {e}")
            return ""

    async def verify_all(self, question_data: dict, teacher_outputs: List[dict]) -> dict:
        qtype = (question_data.get("question_type") or "reasoning").lower()
        if qtype == "math":
            return self._verify_math(question_data["question"], teacher_outputs)
        if qtype == "code":
            return self._verify_code(question_data["question"], teacher_outputs)
        if qtype == "factual":
            return await self._verify_factual(question_data["question"], teacher_outputs)
        return await self._verify_reasoning(question_data["question"], teacher_outputs)

    def _verify_math(self, question: str, teacher_outputs: List[dict]) -> dict:
        expected = None
        # Try to find an expression in the question
        expr_match = re.search(r"([\d\.\+\-\*\/\(\)\s\^]{3,})", question)
        if expr_match:
            expr = expr_match.group(1).replace("^", "**")
            try:
                # Restricted eval: numbers/operators only
                if re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", expr):
                    expected = float(eval(expr, {"__builtins__": {}}, {}))
            except Exception:
                expected = None

        labels = []
        gold_label = "UNVERIFIABLE"
        for t in teacher_outputs:
            num = _extract_number(t.get("answer", ""))
            if expected is not None and num is not None:
                if abs(num - expected) < 1e-3:
                    labels.append({"style": t.get("style"), "label": "correct"})
                else:
                    labels.append({"style": t.get("style"), "label": "incorrect"})
            else:
                labels.append({"style": t.get("style"), "label": "unverifiable"})

        # Pick gold label: prefer expected, else most-common teacher number
        if expected is not None:
            gold_label = str(expected)
        else:
            nums = [_extract_number(t.get("answer", "")) for t in teacher_outputs]
            nums = [n for n in nums if n is not None]
            if nums:
                # majority
                from collections import Counter
                most_common = Counter(nums).most_common(1)[0][0]
                gold_label = str(most_common)
                # relabel against this gold
                for i, n in enumerate([_extract_number(t.get("answer", "")) for t in teacher_outputs]):
                    if n is None:
                        labels[i] = {"style": teacher_outputs[i].get("style"), "label": "unverifiable"}
                    elif abs(n - most_common) < 1e-3:
                        labels[i] = {"style": teacher_outputs[i].get("style"), "label": "correct"}
                    else:
                        labels[i] = {"style": teacher_outputs[i].get("style"), "label": "incorrect"}

        return {"gold_label": gold_label, "labels": labels, "verifier_type": "math"}

    def _verify_code(self, question: str, teacher_outputs: List[dict]) -> dict:
        expected_match = re.search(r"(?:expected output|output should be|returns?)[:\s]+(.+?)(?:\n|$)", question, re.IGNORECASE)
        expected = expected_match.group(1).strip() if expected_match else None

        labels = []
        outputs_seen: List[str] = []
        for t in teacher_outputs:
            code = _extract_code_block(t.get("answer", ""))
            if not code:
                labels.append({"style": t.get("style"), "label": "unverifiable"})
                outputs_seen.append("")
                continue
            try:
                proc = subprocess.run(
                    ["python3", "-c", code],
                    timeout=5,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                stdout = (proc.stdout or "").strip()
                outputs_seen.append(stdout)
                if proc.returncode != 0:
                    labels.append({"style": t.get("style"), "label": "incorrect"})
                elif expected is not None:
                    if expected.strip() == stdout:
                        labels.append({"style": t.get("style"), "label": "correct"})
                    else:
                        labels.append({"style": t.get("style"), "label": "incorrect"})
                else:
                    labels.append({"style": t.get("style"), "label": "unverifiable"})
            except subprocess.TimeoutExpired:
                outputs_seen.append("")
                labels.append({"style": t.get("style"), "label": "incorrect"})
            except Exception:
                outputs_seen.append("")
                labels.append({"style": t.get("style"), "label": "incorrect"})

        # If no explicit expected, use majority-vote on stdout
        if expected is None and outputs_seen:
            from collections import Counter
            non_empty = [o for o in outputs_seen if o]
            if non_empty:
                gold_out = Counter(non_empty).most_common(1)[0][0]
                gold_label = gold_out
                for i, o in enumerate(outputs_seen):
                    if not o:
                        continue
                    if o == gold_out:
                        labels[i] = {"style": teacher_outputs[i].get("style"), "label": "correct"}
                    else:
                        labels[i] = {"style": teacher_outputs[i].get("style"), "label": "incorrect"}
            else:
                gold_label = "UNVERIFIABLE"
        else:
            gold_label = expected or "UNVERIFIABLE"

        return {"gold_label": gold_label, "labels": labels, "verifier_type": "code"}

    async def _verify_factual(self, question: str, teacher_outputs: List[dict]) -> dict:
        context = ""
        if self.faiss_available:
            # Real FAISS retrieval would go here; we noted the index but don't ship
            # an embedding model dependency for retrieval to keep deps light.
            context = ""

        teachers_block = "\n".join(
            f"- {t.get('style')}: {str(t.get('answer', ''))[:300]}"
            for t in teacher_outputs
        )
        system = (
            "You are a precise factual verifier. Given a question and three candidate "
            "answers, return strict JSON of the form: "
            "{\"gold_answer\": str, \"labels\": [{\"style\": str, \"label\": one of correct|incorrect|unverifiable}]}."
        )
        user = f"Question:\n{question}\n\nCandidate answers:\n{teachers_block}"
        if context:
            user = f"Reference context:\n{context}\n\n" + user

        raw = await self._groq_chat(system, user, temperature=0.0, max_tokens=600)
        labels = []
        gold_label = "UNVERIFIABLE"
        try:
            cleaned = raw.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(cleaned[start : end + 1])
                gold_label = str(parsed.get("gold_answer", "UNVERIFIABLE"))
                labels = parsed.get("labels", [])
                # normalise labels list
                styles_seen = {l.get("style"): l for l in labels if isinstance(l, dict)}
                labels = [
                    {
                        "style": t.get("style"),
                        "label": (styles_seen.get(t.get("style"), {}).get("label") or "unverifiable"),
                    }
                    for t in teacher_outputs
                ]
        except Exception:
            labels = [{"style": t.get("style"), "label": "unverifiable"} for t in teacher_outputs]

        return {"gold_label": gold_label, "labels": labels, "verifier_type": "factual"}

    async def _verify_reasoning(self, question: str, teacher_outputs: List[dict]) -> dict:
        teachers_block = "\n".join(
            f"- {t.get('style')}: {str(t.get('answer', ''))[:400]}"
            for t in teacher_outputs
        )
        system = (
            "You are a logic/reasoning grader. Read the question and the three answers. "
            "Return strict JSON: "
            "{\"gold_answer\": str, \"labels\": [{\"style\": str, \"label\": one of correct|incorrect|unverifiable}]}."
            " Pick the most logically sound answer as the gold."
        )
        user = f"Question:\n{question}\n\nCandidate answers:\n{teachers_block}"
        raw = await self._groq_chat(system, user, temperature=0.0, max_tokens=600)
        labels = []
        gold_label = "UNVERIFIABLE"
        try:
            cleaned = raw.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(cleaned[start : end + 1])
                gold_label = str(parsed.get("gold_answer", "UNVERIFIABLE"))
                styles_seen = {
                    l.get("style"): l for l in parsed.get("labels", []) if isinstance(l, dict)
                }
                labels = [
                    {
                        "style": t.get("style"),
                        "label": (styles_seen.get(t.get("style"), {}).get("label") or "unverifiable"),
                    }
                    for t in teacher_outputs
                ]
        except Exception:
            labels = [{"style": t.get("style"), "label": "unverifiable"} for t in teacher_outputs]

        return {"gold_label": gold_label, "labels": labels, "verifier_type": "reasoning"}
