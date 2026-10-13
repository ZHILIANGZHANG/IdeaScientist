"""Model registry (configs/models.yaml, optionally extended by a user YAML) and loading."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import yaml

from ..tokenization import HFTokenizer
from ..utils import PACKAGE_ROOT
from .base import LoopAdapter, ModulePaths
from .hf_decoder import HFDecoderAdapter
from .ouro import OuroAdapter, patch_ouro_for_transformers5
from .raven import RavenAdapter, patch_raven_for_transformers5

ADAPTERS: dict[str, type[LoopAdapter]] = {"ouro": OuroAdapter, "raven": RavenAdapter,
                                         "hf": HFDecoderAdapter}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DEFAULT_MODELS_YAML = PACKAGE_ROOT / "configs" / "models.yaml"


def load_registry(extra_yaml: Path | None = None) -> dict[str, dict[str, Any]]:
    reg = yaml.safe_load(DEFAULT_MODELS_YAML.read_text())
    if extra_yaml is not None:
        reg.update(yaml.safe_load(Path(extra_yaml).read_text()) or {})
    return reg


def model_spec(name: str, extra_yaml: Path | None = None) -> dict[str, Any]:
    reg = load_registry(extra_yaml)
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; known: {sorted(reg)} (add it to a YAML via --models-yaml)")
    spec = dict(reg[name])
    spec["name"] = name
    if spec["adapter"] not in ADAPTERS:
        raise ValueError(f"{name}: adapter must be one of {sorted(ADAPTERS)}")
    return spec


def module_paths(spec: dict[str, Any]) -> ModulePaths:
    base = ADAPTERS[spec["adapter"]].default_paths
    overrides = spec.get("module_paths") or {}
    unknown = set(overrides) - set(ModulePaths.__dataclass_fields__)
    if unknown:
        raise ValueError(f"{spec['name']}: unknown module_paths fields {sorted(unknown)}")
    return ModulePaths(**{**base.__dict__, **overrides})


def _dtype_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


def load_adapter(spec: dict[str, Any], device: str, dtype: str | None = None,
                 model_path: str | None = None) -> LoopAdapter:
    torch_dtype = DTYPES[dtype or spec.get("dtype", "bfloat16")]
    cls = ADAPTERS[spec["adapter"]]
    options = dict(spec.get("options") or {})
    if spec.get("builder"):  # in-process builder (fake models for tests / smoke)
        mod_name, fn_name = spec["builder"].split(":")
        model, tok = getattr(importlib.import_module(mod_name), fn_name)()
        model = model.to(device=device, dtype=torch_dtype)
    else:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = model_path or spec["hf_id"]
        trc = bool(spec.get("trust_remote_code", False))
        if spec["adapter"] == "ouro":
            patch_ouro_for_transformers5(model_id)
        if spec["adapter"] == "raven":
            patch_raven_for_transformers5(model_id)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trc)
        if spec["adapter"] == "ouro" and not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        kwargs = _dtype_kwargs(torch_dtype)
        if spec.get("attn_implementation"):
            kwargs["attn_implementation"] = spec["attn_implementation"]
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=trc,
                                                     **kwargs).to(device)
        tok = HFTokenizer(AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc), spec["name"])
    return cls(model, tok, name=spec["name"], paths=module_paths(spec),
               n_loops_trained=int(spec.get("n_loops_trained", 1)), device=device, options=options)
