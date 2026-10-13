"""Small shared helpers: seeding, threads, metadata, JSON/JSONL I/O, resumable shards, logging."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent


def set_threads(n: int) -> None:
    """Cap CPU threads for torch / BLAS (call before heavy work)."""
    import torch

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def git_hash() -> str:
    """Commit hash of the checkout containing this package (+'-dirty' if modified)."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--", "."],
            cwd=PACKAGE_ROOT, stderr=subprocess.DEVNULL, text=True,
        )
        code_changes = [ln for ln in status.splitlines() if "results" not in ln]
        return head + ("-dirty" if code_changes else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Self-describing header written next to every output file."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "numpy", "sklearn", "statsmodels"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = "missing"
    return {
        "git_hash": git_hash(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "versions": versions,
        "config": config,
    }


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default))
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomic JSONL write (tmp file + rename) so partial shards never look finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_default) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl_glob(directory: Path, pattern: str) -> Iterator[dict[str, Any]]:
    for p in sorted(Path(directory).glob(pattern)):
        yield from read_jsonl(p)


def shard_done(path: Path) -> bool:
    return path.exists() and path.with_suffix(path.suffix + ".done").exists()


def mark_done(path: Path, meta: dict[str, Any]) -> None:
    write_json(path.with_suffix(path.suffix + ".done"), meta)


def get_logger(name: str, logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        target = str(logfile.resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target
                   for h in logger.handlers):
            fh = logging.FileHandler(target)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


def parse_layer_selector(selector: str | list[int], n_layers: int) -> list[int]:
    """'all' | 'every:K' | 'every:K+last' | explicit list -> sorted layer indices."""
    if isinstance(selector, list):
        return sorted({int(i) % n_layers for i in selector})
    if selector == "all":
        return list(range(n_layers))
    if selector.startswith("every:"):
        spec = selector[len("every:"):]
        add_last = spec.endswith("+last")
        k = int(spec.replace("+last", ""))
        layers = set(range(0, n_layers, k))
        if add_last:
            layers.add(n_layers - 1)
        return sorted(layers)
    raise ValueError(f"bad layer selector {selector!r}; use 'all', 'every:K[+last]' or a list")
