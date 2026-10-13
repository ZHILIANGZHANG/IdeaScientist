"""Shared run context: configuration, difficulty levels, stimuli, prompt preparation."""
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
from ..utils import PACKAGE_ROOT, read_json, run_metadata, write_json

EXPERIMENTS_YAML = PACKAGE_ROOT / "configs" / "experiments.yaml"
EXPERIMENT_NAMES = ("calibrate", "sanity", "B", "A1", "A2", "H3", "C")
CALIBRATION_FILE = "calibration.json"


def load_profile(profile: str, path: Path | None = None) -> dict[str, Any]:
    profiles = yaml.safe_load((path or EXPERIMENTS_YAML).read_text())
    if profile not in profiles:
        raise KeyError(f"unknown profile {profile!r}; known {sorted(profiles)}")
    return profiles[profile]


def per_family(value: Any, family: str, default: int) -> int:
    """`contexts_per_cell` may be an int or a {family: int} mapping."""
    if isinstance(value, dict):
        return int(value.get(family, default))
    return int(value if value is not None else default)


@dataclass
class Prepared:
    ids: list[int]
    fact_pos: list[int]  # last token of each fact value
    ctx_pos: list[int]  # all tokens of the fact block (header and question excluded)
    candidates: list[str]
    answer: int
    gist_pos: list[int] | None = None  # last token of each stated category (explicit families)


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

    @property
    def levels(self) -> list[dict[str, Any]]:
        return list(self.profile["levels"])

    def level(self, name: str) -> dict[str, Any]:
        return next(lv for lv in self.levels if lv["name"] == name)

    def exp_cfg(self, exp: str) -> dict[str, Any]:
        """Profile settings for one experiment, with model-level overrides (models.yaml)."""
        cfg = dict(self.profile.get(exp, {}))
        cfg.update((self.spec.get("experiments") or {}).get(exp, {}))
        return cfg

    def exp_dir(self, exp: str) -> Path:
        d = self.out_dir / exp
        d.mkdir(parents=True, exist_ok=True)
        return d

    def calibration(self) -> dict[str, Any] | None:
        p = self.out_dir / CALIBRATION_FILE
        return read_json(p) if p.exists() else None

    def meta(self, exp: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        cal = self.calibration()
        return run_metadata({"model": self.spec, "profile": self.profile_name, "experiment": exp,
                             "exp_config": self.exp_cfg(exp), "chat": self.chat,
                             "adapter": self.adapter.describe(),
                             "calibrated_levels": cal["selected"] if cal else None, **(extra or {})})

    def stimuli(self) -> dict[tuple[str, str], list[dict]]:
        if self._stimuli is None:
            p = self.profile
            self._stimuli = build_stimuli(self.adapter.tok, self.out_dir, list(p["families"]), self.levels,
                                          int(p["contexts_per_cell"]), self.seed, chat=self.chat,
                                          max_context=self.adapter.max_context())
        return self._stimuli

    def selected_levels(self, exp: str, family: str) -> list[str]:
        """Explicit `levels` in the experiment config win; otherwise the calibrated selection
        (results/<model>/calibration.json); otherwise every profile level."""
        cfg = self.exp_cfg(exp)
        if cfg.get("levels"):
            return list(cfg["levels"])
        cal = self.calibration()
        if cal and family in cal["selected"]:
            return list(cal["selected"][family])
        return [lv["name"] for lv in self.levels]

    def cells(self, exp: str) -> list[tuple[str, dict[str, Any], list[dict]]]:
        """(family, level, first-k instances) for an experiment's configured cells; levels
        that did not fit the context window are skipped with a log line."""
        cfg = self.exp_cfg(exp)
        st = self.stimuli()
        out = []
        for fam in cfg.get("families", self.profile["families"]):
            k = per_family(cfg.get("contexts_per_cell"), fam, int(self.profile["contexts_per_cell"]))
            for name in self.selected_levels(exp, fam):
                if (fam, name) not in st:
                    self.logger.warning("%s: no stimuli for %s/%s (did not fit the context?)", exp, fam, name)
                    continue
                out.append((fam, self.level(name), st[(fam, name)][:k]))
        return out

    def prepare(self, prompt: Prompt) -> Prepared:
        tok = self.adapter.tok
        ids, offs = tok.encode_offsets(prompt.text, add_special=not self.chat)
        fact_pos = [token_span(offs, s, e)[-1] for s, e in prompt.fact_spans]
        gist_pos = [token_span(offs, s, e)[-1] for s, e in prompt.gist_spans] if prompt.gist_spans else None
        ctx_pos = token_span(offs, *prompt.context_span) if prompt.context_span is not None else []
        return Prepared(ids=ids, fact_pos=fact_pos, ctx_pos=ctx_pos, candidates=prompt.candidates,
                        answer=prompt.answer, gist_pos=gist_pos)

    def tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], device=self.adapter.device)

    def loops_eval(self) -> list[int]:
        return [int(t) for t in self.spec.get("loops_eval", [1])]

    def write_summary(self, exp: str, summary: dict[str, Any]) -> None:
        write_json(self.exp_dir(exp) / "summary.json", {"meta": self.meta(exp), **summary})


def cell_fields(inst: dict) -> dict[str, Any]:
    """Columns identifying an instance's cell, shared by all per-item rows."""
    return {"item_id": inst["id"], "context_id": inst["id"], "family": inst["family"],
            "level": inst["level"], "N": inst["N"], "fillers_per_fact": inst["fillers_per_fact"],
            "distance": inst["distance"], "query_pos": inst["query_idx"],
            "query_quartile": inst["query_quartile"]}
