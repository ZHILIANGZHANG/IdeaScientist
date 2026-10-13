"""Turn a stimulus instance (JSON dict) into prompt text + candidate strings + char spans.

Layout (plain completion):
    {header}\n{fact_1}[ {filler} ...]\n...{fact_N}...\n{question}\n{answer_prefix}
and the candidates (each with a leading space) are scored as the continuation.
With `chat=True` the header/facts/question go into one user turn of the tokenizer's chat
template (thinking disabled where supported); the answer prefix follows the generation
prompt. Explicit families state the coarse attribute in the fact itself, so detail and
gist queries are both pure retrieval of stated information.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..tokenization import TextTokenizer
from .generate import base_family, category_of, is_explicit
from .vocab import FILLERS

TEMPLATES: dict[str, dict[str, str]] = {
    "box": {
        "header": "Each box below contains one object.",
        "fact": "Box {label} contains the {value}.",
        "fact_explicit": "Box {label} contains the {value}, a kind of {gist}.",
        "detail_q": "Question: Which exact object is in Box {label}?",
        "detail_a": "Answer: Box {label} contains the",
        "gist_q": "Question: What kind of object is in Box {label}?",
        "gist_a": "Answer: It is a kind of",
        "knowledge": "The {value} is a kind of",
    },
    "codes": {
        "header": "Each person below has a secret four-digit code.",
        "fact": "The code of {label} is {value}.",
        "fact_explicit": "The code of {label} is {value}, in the {gist}.",
        "detail_q": "Question: What is the code of {label}?",
        "detail_a": "Answer: The code of {label} is",
        "gist_q": "Question: In which range of thousands is the code of {label}?",
        "gist_a": "Answer: It is in the",
        "knowledge": "The number {value} is in the",
    },
}
INFO_TYPES = ("detail", "gist")


@dataclass
class Prompt:
    text: str
    candidates: list[str]  # continuation strings, each with a leading space
    answer: int  # index of the correct candidate
    fact_spans: list[tuple[int, int]]  # char span of each fact VALUE (empty w/o context)
    context_span: tuple[int, int] | None  # char span of the fact block (None w/o context)
    gist_spans: list[tuple[int, int]] | None = None  # stated category spans (explicit only)


def candidate_strings(inst: dict, info_type: str) -> list[str]:
    return [f" {c}" for c in inst[info_type]["candidates"]]


def _fact_block(inst: dict, values: list[str], start: int) -> tuple[str, list, list]:
    tpl = TEMPLATES[base_family(inst["family"])]
    explicit = is_explicit(inst["family"])
    text, spans, gspans = "", [], []
    for k, (lab, val) in enumerate(zip(inst["labels"], values)):
        gist = category_of(inst, val)
        line = (tpl["fact_explicit"] if explicit else tpl["fact"]).format(label=lab, value=val, gist=gist)
        v_off = line.index(f" {val}") + 1
        spans.append((start + len(text) + v_off, start + len(text) + v_off + len(val)))
        if explicit:
            g_off = line.rindex(gist)
            gspans.append((start + len(text) + g_off, start + len(text) + g_off + len(gist)))
        fill = "".join(" " + FILLERS[f] for f in inst["fillers"][k]) if inst.get("fillers") else ""
        text += line + fill + "\n"
    return text, spans, gspans


def render(inst: dict, info_type: str, *, values: list[str] | None = None,
           with_context: bool = True, chat_tok: TextTokenizer | None = None) -> Prompt:
    """Render one query of an instance. `values` overrides the fact values (minimal pairs /
    different-input control); labels, fillers and token positions stay identical."""
    if info_type not in INFO_TYPES:
        raise ValueError(info_type)
    tpl = TEMPLATES[base_family(inst["family"])]
    values = list(values if values is not None else inst["values"])
    label = inst["labels"][inst["query_idx"]]
    user = tpl["header"] + "\n"
    spans, gspans, ctx_span = [], [], None
    if with_context:
        block, spans, gspans = _fact_block(inst, values, len(user))
        ctx_span = (len(user), len(user) + len(block) - 1)
        user += block
    user += tpl[f"{info_type}_q"].format(label=label)
    answer_prefix = tpl[f"{info_type}_a"].format(label=label)
    if chat_tok is None:
        text, shift = user + "\n" + answer_prefix, 0
    else:
        formatted = chat_tok.chat_prompt(user)
        shift = formatted.index(user)
        text = formatted + answer_prefix

    def sh(xs):
        return [(s + shift, e + shift) for s, e in xs]

    return Prompt(text=text, candidates=candidate_strings(inst, info_type),
                  answer=int(inst[info_type]["answer"]), fact_spans=sh(spans),
                  context_span=None if ctx_span is None else (ctx_span[0] + shift, ctx_span[1] + shift),
                  gist_spans=sh(gspans) if gspans else None)


def render_context_only(inst: dict, *, values: list[str] | None = None) -> Prompt:
    """Header + facts only (probe feature extraction at fact positions)."""
    tpl = TEMPLATES[base_family(inst["family"])]
    values = list(values if values is not None else inst["values"])
    head = tpl["header"] + "\n"
    block, spans, gspans = _fact_block(inst, values, len(head))
    text = (head + block).rstrip("\n")
    return Prompt(text=text, candidates=[], answer=-1, fact_spans=spans,
                  context_span=(len(head), len(text)), gist_spans=gspans or None)


def render_knowledge(inst: dict) -> Prompt:
    """No-context category-knowledge control for the QUERIED value with the instance's own
    gist candidates: 'The zebra is a kind of' -> animal? / 'The number 4821 is in the' -> 4000s?"""
    tpl = TEMPLATES[base_family(inst["family"])]
    value = inst["values"][inst["query_idx"]]
    return Prompt(text=tpl["knowledge"].format(value=value), candidates=candidate_strings(inst, "gist"),
                  answer=int(inst["gist"]["answer"]), fact_spans=[], context_span=None)
