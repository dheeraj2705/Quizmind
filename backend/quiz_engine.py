"""
quiz_engine.py — Groq-powered quiz generation and answer evaluation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from groq import Groq
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Question(BaseModel):
    id: int
    type: str  # "mcq" | "true_false" | "short_answer"
    question: str
    options: List[str] = Field(default_factory=list)
    correct_answer: str
    explanation: str
    difficulty: str


class ScoreResult(BaseModel):
    correct: bool
    score: float       # 0.0 – 1.0
    feedback: str


# ---------------------------------------------------------------------------
# System / user prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert educator and quiz creator. Your job is to create \
high-quality quiz questions strictly based on the provided study material.

Rules:
1. Only use information explicitly present in the provided context
2. Do NOT add outside knowledge
3. Questions must test genuine understanding, not just memorization
4. Adjust complexity precisely to the difficulty level specified
5. Return ONLY valid JSON, no markdown, no explanation

Difficulty guidelines:
- easy: recall facts, basic comprehension
- medium: application, inference, connecting concepts
- hard: analysis, evaluation, edge cases, nuanced understanding"""


def _build_user_prompt(
    context: str,
    num_questions: int,
    difficulty: str,
    question_types: List[str],
) -> str:
    type_labels = {
        "mcq": "Multiple Choice (MCQ)",
        "true_false": "True/False",
        "short_answer": "Short Answer",
    }
    readable_types = ", ".join(type_labels.get(t, t) for t in question_types)

    return f"""Based ONLY on the following study material, generate {num_questions} \
quiz questions at {difficulty} difficulty.

Include these question types: {readable_types}

STUDY MATERIAL:
{context}

Return a JSON array with this exact structure:
[
  {{
    "id": 1,
    "type": "mcq",
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "A",
    "explanation": "Brief explanation of why this is correct",
    "difficulty": "{difficulty}"
  }},
  {{
    "id": 2,
    "type": "true_false",
    "question": "...",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "...",
    "difficulty": "{difficulty}"
  }},
  {{
    "id": 3,
    "type": "short_answer",
    "question": "...",
    "options": [],
    "correct_answer": "Expected answer keywords: ...",
    "explanation": "...",
    "difficulty": "{difficulty}"
  }}
]

Return ONLY the JSON array, nothing else."""


SCORE_SYSTEM_PROMPT = """You are an impartial grader evaluating short-answer quiz \
responses against an expected answer. Return ONLY valid JSON with no markdown."""


def _build_score_prompt(question: str, expected: str, user_answer: str) -> str:
    return f"""Question: {question}

Expected answer / key concepts: {expected}

Student answer: {user_answer}

Evaluate the student's answer and return JSON in exactly this format:
{{
  "correct": true or false,
  "score": 0.0 to 1.0 (partial credit is fine),
  "feedback": "One or two sentences of constructive feedback"
}}

Return ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QuizEngine:
    """Generate quizzes and evaluate short answers using the Groq API."""

    MODEL = "llama-3.3-70b-versatile"
    MAX_RETRIES = 2

    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Quiz generation
    # ------------------------------------------------------------------

    def generate_quiz(
        self,
        context_chunks: List[str],
        num_questions: int,
        difficulty: str,
        question_types: List[str],
    ) -> List[Question]:
        """Generate quiz questions grounded in *context_chunks*.

        Args:
            context_chunks: Retrieved relevant text chunks.
            num_questions:  Target question count (5–20).
            difficulty:     "easy" | "medium" | "hard".
            question_types: Subset of ["mcq", "true_false", "short_answer"].

        Returns:
            Parsed list of Question objects.

        Raises:
            ValueError: When the LLM cannot produce parseable JSON after retries.
        """
        if not context_chunks:
            raise ValueError("No study material found. Please upload documents first.")

        context = "\n\n---\n\n".join(context_chunks)
        # Trim context to avoid exceeding context window (~8 k tokens ≈ 32 k chars)
        if len(context) > 32_000:
            context = context[:32_000] + "\n\n[... content trimmed ...]"

        user_prompt = _build_user_prompt(context, num_questions, difficulty, question_types)

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                raw = self._call_groq(SYSTEM_PROMPT, user_prompt)
                questions = self._parse_questions(raw)
                if questions:
                    return questions
                raise ValueError("LLM returned an empty question list.")
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    # Nudge the model on retry
                    user_prompt = user_prompt + "\n\nIMPORTANT: Return ONLY the raw JSON array. No prose, no code fences."

        raise ValueError(
            f"Quiz generation failed after {self.MAX_RETRIES + 1} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Answer scoring
    # ------------------------------------------------------------------

    def score_answer(
        self,
        question: Dict[str, Any],
        user_answer: str,
    ) -> ScoreResult:
        """Score a short-answer response using the Groq LLM.

        For MCQ / True-False questions this is handled client-side;
        this method is specifically for open-ended responses.

        Args:
            question:    The original question dict (must have "question" and
                         "correct_answer" keys).
            user_answer: The student's free-text answer.

        Returns:
            ScoreResult with correct flag, 0–1 score, and feedback string.
        """
        q_text = question.get("question", "")
        expected = question.get("correct_answer", "")

        if not user_answer.strip():
            return ScoreResult(
                correct=False,
                score=0.0,
                feedback="No answer was provided.",
            )

        prompt = _build_score_prompt(q_text, expected, user_answer)

        try:
            raw = self._call_groq(SCORE_SYSTEM_PROMPT, prompt, max_tokens=256)
            data = self._extract_json_object(raw)
            return ScoreResult(
                correct=bool(data.get("correct", False)),
                score=float(data.get("score", 0.0)),
                feedback=str(data.get("feedback", "")),
            )
        except Exception as exc:
            # Fallback: simple keyword matching
            return self._keyword_score(expected, user_answer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Call Groq chat completion and return the raw text response."""
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response.choices[0].message.content or ""

    def _parse_questions(self, raw: str) -> List[Question]:
        """Extract and validate a JSON array of questions from *raw* text."""
        data = self._extract_json_array(raw)
        questions: List[Question] = []
        for i, item in enumerate(data):
            try:
                # Ensure id is set
                item.setdefault("id", i + 1)
                item.setdefault("options", [])
                item.setdefault("explanation", "")
                questions.append(Question(**item))
            except Exception:
                continue
        return questions

    @staticmethod
    def _extract_json_array(raw: str) -> List[Dict]:
        """Robustly extract the first JSON array from *raw*."""
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Try direct parse first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Find the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract a JSON array from LLM response: {raw[:300]!r}")

    @staticmethod
    def _extract_json_object(raw: str) -> Dict:
        """Robustly extract the first JSON object from *raw*."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract a JSON object: {raw[:200]!r}")

    @staticmethod
    def _keyword_score(expected: str, user_answer: str) -> ScoreResult:
        """Fallback scorer: count keyword overlap."""
        keywords = re.findall(r"\b\w{4,}\b", expected.lower())
        if not keywords:
            return ScoreResult(correct=False, score=0.0, feedback="Could not evaluate answer.")

        answer_lower = user_answer.lower()
        hits = sum(1 for kw in keywords if kw in answer_lower)
        score = min(hits / len(keywords), 1.0)
        correct = score >= 0.5

        return ScoreResult(
            correct=correct,
            score=round(score, 2),
            feedback=(
                "Good answer! You covered the key concepts."
                if correct
                else "Your answer is missing some key concepts from the material."
            ),
        )
