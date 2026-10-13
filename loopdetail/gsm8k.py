"""GSM8K helpers: data loading, prompts, answer extraction and automatic operand tagging.

Detail-error tagging: every 'expr = result' segment in the generation is parsed; each
numeric operand of `expr` is labelled
  problem   - occurs in the problem text (digits or number words),
  computed  - equals an earlier result, or occurs earlier in the generation,
  constant  - a small common constant (1, 2, 10, 12, 24, 60, 100, 1000, ...),
  unexplained otherwise.
detail_error = at least one unexplained operand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?%?"
_EXPR_CHARS = set("0123456789.,$%()+-*/x×÷ ")
_OPERATOR = re.compile(r"\d[%)]?\s*[-+*/x×÷]\s*[($]?\d")
_RESULT = re.compile(rf"=\s*({NUM})")
_NUM_RE = re.compile(NUM)
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
         "half": 0.5, "twice": 2, "double": 2, "triple": 3, "quarter": 0.25, "dozen": 12}
CONSTANTS = {0.5, 1, 2, 3, 4, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000}


def to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").rstrip("%").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def numbers_in(text: str) -> set[float]:
    vals = {v for v in (to_float(m) for m in _NUM_RE.findall(text)) if v is not None}
    for m in _NUM_RE.findall(text):
        if m.endswith("%") and (v := to_float(m)) is not None:
            vals.add(v / 100)
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORDS:
            vals.add(float(WORDS[w]))
    return vals


def _close(a: float, b: set[float]) -> bool:
    return any(abs(a - x) <= 1e-6 * max(1.0, abs(x)) for x in b)


def tag_operands(problem: str, generation: str) -> dict[str, Any]:
    problem_nums = numbers_in(problem)
    computed: set[float] = set()
    equations = []
    for m in _RESULT.finditer(generation):
        start = m.start()
        while start > 0 and generation[start - 1] in _EXPR_CHARS:
            start -= 1
        expr, res = generation[start: m.start()], m.group(1)
        if not _OPERATOR.search(expr):
            continue
        before = numbers_in(generation[:start])
        ops = []
        for tok in _NUM_RE.findall(expr):
            v = to_float(tok)
            if v is None:
                continue
            if _close(v, problem_nums):
                src = "problem"
            elif _close(v, computed) or _close(v, before):
                src = "computed"
            elif _close(v, set(map(float, CONSTANTS))):
                src = "constant"
            else:
                src = "unexplained"
            ops.append({"value": v, "source": src})
        r = to_float(res)
        if r is not None:
            computed.add(r)
        equations.append({"expr": expr.strip(), "result": res, "operands": ops})
    unexplained = [o["value"] for e in equations for o in e["operands"] if o["source"] == "unexplained"]
    return {"equations": equations, "n_equations": len(equations),
            "unexplained_operands": unexplained, "detail_error": int(bool(unexplained))}


def extract_answer(text: str) -> float | None:
    text = text.split("\nQuestion:")[0]
    for pat in (r"####\s*(" + NUM + ")", r"\\boxed\{\s*(" + NUM + r")\s*\}",
                r"answer is[:\s]*(" + NUM + ")"):
        m = re.findall(pat, text)
        if m:
            return to_float(m[-1])
    nums = _NUM_RE.findall(text)
    return to_float(nums[-1]) if nums else None


def gold_answer(answer_field: str) -> float:
    v = to_float(answer_field.split("####")[-1].strip())
    if v is None:
        raise ValueError(f"cannot parse GSM8K gold answer {answer_field!r}")
    return v


def load_gsm8k(path: str | None, split: str = "test") -> list[dict[str, str]]:
    """Local GSM8K jsonl (fields question/answer) or HF `openai/gsm8k` (main)."""
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / f"{split}.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    from datasets import load_dataset

    return [dict(r) for r in load_dataset("openai/gsm8k", "main", split=split)]


def strip_calc(solution: str) -> str:
    """Remove GSM8K calculator annotations <<a*b=c>>."""
    return re.sub(r"<<[^>]*>>", "", solution)


def build_prompt(question: str, shots: list[dict[str, str]]) -> str:
    """Plain few-shot prompt for base models (solutions end with '#### <answer>')."""
    demo = "".join(f"Question: {s['question']}\nAnswer: {strip_calc(s['answer'])}\n\n" for s in shots)
    return demo + f"Question: {question}\nAnswer:"


CHAT_INSTRUCTION = ("Solve the following math problem step by step. Write the final numeric "
                    "answer on the last line in the form '#### <answer>'.\n\n")
