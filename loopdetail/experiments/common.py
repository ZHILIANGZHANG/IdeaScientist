"""Shared run context: configuration, stimuli, prompt preparation, output locations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models.base import LoopAdapter
from ..stimuli.build import build_stimuli
from ..stimuli.render import Prompt
from ..tokenization import token_span
from ..utils import PACKAGE_ROOT, run_metadata, write_json

EXPERIMENTS_YAML = PACKAGE_ROOT / "configs" / "experiments.yaml"
EXPERIMENT_NAMES = ("sanity", "B", "A1", "A2", "H3", "C")


def load_profile(profile: str, path: Path | None = None) -> dict[str, Any]:
    profiles = yaml.safe_load((path or EXPERIMENTS_YAML).read_text())
    if profile not in profiles:
        raise KeyError(f"unknown profile {profile!r}; known {sorted(profiles)}")
    return profiles[profile]


@dataclass
class Prepared:
    ids: list[int]
    fact_pos: list[int]  # last token of each fact value
    ctx_pos: list[int]  # all tokens of the fact block (header and question excluded)
    candidates: list[str]
    answer: int


@dataclass
class RunContext:
    adapter: LoopAdapter
    spec: dict[str, Any]
    profile: dict[str, Any]
    profile_name: str
    out_dir: Path  # results/<model>
    chat: bool
    logger: logging.Logger
    force: bool = False
    _stimuli: dict | None = field(default=None, repr=False)

    @property
    def seed(self) -> int:
        return int(self.profile["seed"])

    def exp_cfg(self, exp: str) -> dict[str, Any]:
        """Profile settings for one experiment, with model-level overrides (models.yaml)."""
        cfg = dict(self.profile.get(exp, {}))
        cfg.update((self.spec.get("experiments") or {}).get(exp, {}))
        return cfg

    def exp_dir(self, exp: str) -> Path:
        d = self.out_dir / exp
        d.mkdir(parents=True, exist_ok=True)
        return d

    def meta(self, exp: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return run_metadata({"model": self.spec, "profile": self.profile_name, "experiment": exp,
                             "exp_config": self.exp_cfg(exp), "chat": self.chat,
                             "adapter": self.adapter.describe(), **(extra or {})})

    def stimuli(self) -> dict[tuple[str, int], list[dict]]:
        if self._stimuli is None:
            p = self.profile
            self._stimuli = build_stimuli(self.adapter.tok, self.out_dir, list(p["families"]),
                                          list(p["n_values"]), int(p["contexts_per_cell"]),
                                          self.seed, chat=self.chat)
        return self._stimuli

    def cells(self, exp: str) -> list[tuple[str, int, list[dict]]]:
        """(family, N, first-k instances) for an experiment's configured cells."""
        cfg = self.exp_cfg(exp)
        fams = cfg.get("families", self.profile["families"])
        ns = cfg.get("n_values", self.profile["n_values"])
        k = int(cfg.get("contexts_per_cell", self.profile["contexts_per_cell"]))
        st = self.stimuli()
        return [(f, N, st[(f, N)][:k]) for f in fams for N in ns]

    def prepare(self, prompt: Prompt) -> Prepared:
        tok = self.adapter.tok
        ids, offs = tok.encode_offsets(prompt.text, add_special=not self.chat)
        fact_pos = [token_span(offs, s, e)[-1] for s, e in prompt.fact_spans]
        ctx_pos: list[int] = []
        if prompt.context_span is not None:
            ctx_pos = token_span(offs, *prompt.context_span)
        return Prepared(ids=ids, fact_pos=fact_pos, ctx_pos=ctx_pos,
                        candidates=prompt.candidates, answer=prompt.answer)

    def tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], device=self.adapter.device)

    def loops_eval(self) -> list[int]:
        return [int(t) for t in self.spec.get("loops_eval", [1])]

    def write_summary(self, exp: str, summary: dict[str, Any]) -> None:
        write_json(self.exp_dir(exp) / "summary.json", {"meta": self.meta(exp), **summary})
