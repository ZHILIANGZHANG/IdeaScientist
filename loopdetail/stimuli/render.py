"""Turn a stimulus instance (JSON dict) into prompt text + candidate strings + char spans.

Layout (plain completion):
    {header}\n{fact_1}\n...{fact_N}\n{question}\n{answer_prefix}
and the candidates (each with a leading space) are scored as the continuation.
With `chat=True` the header/facts/question go into a single user turn of the tokenizer's
chat template (thinking disabled where supported) and the answer prefix is appended
after the generation prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..tokenization import TextTokenizer

TEMPLATES: dict[str, dict[str, str]] = {
    "box": {
        "header": "Each box below contains one object.",
        "fact": "Box {label} contains the {value}.",
        "detail_q": "Question: Which exact object is in Box {label}?",
        "detail_a": "Answer: Box {label} contains the",
        "gist_q": "Question: What kind of object is in Box {label}?",
        "gist_a": "Answer: It is a kind of",
    },
    "codes": {
        "header": "Each person below has a secret four-digit code.",
        "fact": "The code of {label} is {value}.",
        "detail_q": "Question: What is the code of {label}?",
        "detail_a": "Answer: The code of {label} is",
        "gist_q": "Question: In which range of thousands is the code of {label}?",
        "gist_a": "Answer: It is in the",
    },
}
INFO_TYPES = ("detail", "gist")


@dataclass
class Prompt:
    text: str
    candidates: list[str]  # continuation strings, each with a leading space
    answer: int  # index of the correct candidate
    fact_spans: list[tuple[int, int]]  # char span of each fact VALUE in `text` (empty w/o context)
    context_span: tuple[int, int] | None  # char span of the fact block (None w/o context)


def candidate_strings(inst: dict, info_type: str) -> list[str]:
    return [f" {c}" for c in inst[info_type]["candidates"]]


def render(inst: dict, info_type: str, *, values: list[str] | None = None,
           with_context: bool = True, chat_tok: TextTokenizer | None = None) -> Prompt:
    """Render one query of an instance. `values` overrides the fact values (minimal pairs /
    different-input control); labels and therefore token positions stay identical."""
    if info_type not in INFO_TYPES:
        raise ValueError(info_type)
    tpl = TEMPLATES[inst["family"]]
    values = list(values if values is not None else inst["values"])
    label = inst["labels"][inst["query_idx"]]
    user = tpl["header"] + "\n"
    spans: list[tuple[int, int]] = []
    ctx_span = None
    if with_context:
        ctx_start = len(user)
        for lab, val in zip(inst["labels"], values):
            line = tpl["fact"].format(label=lab, value=val)
            v_off = line.rindex(val)
            spans.append((len(user) + v_off, len(user) + v_off + len(val)))
            user += line + "\n"
        ctx_span = (ctx_start, len(user) - 1)
    user += tpl[f"{info_type}_q"].format(label=label)
    answer_prefix = tpl[f"{info_type}_a"].format(label=label)
    if chat_tok is None:
        text = user + "\n" + answer_prefix
        shift = 0
    else:
        formatted = chat_tok.chat_prompt(user)
        shift = formatted.index(user)
        text = formatted + answer_prefix
    spans = [(s + shift, e + shift) for s, e in spans]
    if ctx_span is not None:
        ctx_span = (ctx_span[0] + shift, ctx_span[1] + shift)
    return Prompt(text=text, candidates=candidate_strings(inst, info_type),
                  answer=int(inst[info_type]["answer"]), fact_spans=spans, context_span=ctx_span)


def render_context_only(inst: dict, *, values: list[str] | None = None) -> Prompt:
    """Header + facts only (used for probe feature extraction at fact positions)."""
    tpl = TEMPLATES[inst["family"]]
    values = list(values if values is not None else inst["values"])
    text = tpl["header"] + "\n"
    spans = []
    start = len(text)
    for lab, val in zip(inst["labels"], values):
        line = tpl["fact"].format(label=lab, value=val)
        v_off = line.rindex(val)
        spans.append((len(text) + v_off, len(text) + v_off + len(val)))
        text += line + "\n"
    return Prompt(text=text.rstrip("\n"), candidates=[], answer=-1, fact_spans=spans,
                  context_span=(start, len(text) - 1))
