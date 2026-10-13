"""Command line: run / inspect / stimuli / analyze / agreement."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .utils import PACKAGE_ROOT, get_logger, seed_everything, set_threads, write_json

DEFAULT_RESULTS = PACKAGE_ROOT / "results"


def _load(args, logger=None):
    from .models.registry import load_adapter, model_spec

    spec = model_spec(args.model, args.models_yaml)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ad = load_adapter(spec, device, args.dtype, args.model_path)
    if logger:
        logger.info("loaded %s on %s in %.1fs: %s", args.model, device, time.time() - t0, ad.describe())
    return spec, ad


def cmd_run(args) -> None:
    from .experiments import calibrate, exp_a1, exp_a2, exp_b, exp_c, exp_h3, sanity
    from .experiments.common import EXPERIMENT_NAMES, RunContext, load_profile

    exps = [e.strip() for e in args.exp.split(",") if e.strip()]
    bad = set(exps) - set(EXPERIMENT_NAMES)
    if bad:
        raise SystemExit(f"unknown experiments {sorted(bad)}; choose from {EXPERIMENT_NAMES}")
    profile_name = "smoke" if args.smoke else args.profile
    out_dir = Path(args.out) / args.model
    if args.smoke:
        out_dir = Path(args.out) / "smoke" / args.model
    logger = get_logger("loopdetail", out_dir / "run.log")
    seed_everything(0)
    spec, ad = _load(args, logger)
    chat = args.chat if args.chat is not None else spec.get("prompt_format", "plain") == "chat"
    profile = load_profile(profile_name, args.experiments_yaml)
    if args.gsm8k_path:
        profile.setdefault("C", {})["path"] = args.gsm8k_path
    ctx = RunContext(adapter=ad, spec=spec, profile=profile, profile_name=profile_name, out_dir=out_dir, chat=chat, logger=logger, force=args.force)
    runners = {"B": exp_b.run, "A1": exp_a1.run, "A2": exp_a2.run, "H3": exp_h3.run, "C": exp_c.run}
    for exp in exps:
        t0 = time.time()
        logger.info("=== %s (%s) ===", exp, profile_name)
        if exp == "calibrate":
            cal = calibrate.run(ctx)
            logger.info("calibrated levels: %s", cal["selected"])
        elif exp == "sanity":
            report = sanity.run(ctx, skip_repro=args.skip_repro)
            logger.info("sanity overall: %s", report["overall"])
        else:
            sanity.assert_sanity_passed(ctx)
            runners[exp](ctx)
        logger.info("=== %s finished in %.1f min ===", exp, (time.time() - t0) / 60)
    if args.analyze:
        from .analysis.report import analyze

        analyze(out_dir)


def cmd_inspect(args) -> None:
    from .models.inspect_model import inspect_adapter

    _, ad = _load(args)
    results = inspect_adapter(ad, args.loops, check_generation=not args.no_generation)
    if args.out_json:
        write_json(Path(args.out_json), results)
    if any(r["status"] == "FAIL" for r in results):
        raise SystemExit("inspect: at least one check FAILED")


def cmd_stimuli(args) -> None:
    from .experiments.common import load_profile
    from .models.registry import model_spec
    from .stimuli.build import build_stimuli
    from .tokenization import HFTokenizer

    spec = model_spec(args.model, args.models_yaml)
    prof = load_profile("smoke" if args.smoke else args.profile, args.experiments_yaml)
    if spec.get("builder"):
        from .tests.fakes import fake_tokenizer

        tok = fake_tokenizer()
    else:
        from transformers import AutoTokenizer

        tok = HFTokenizer(AutoTokenizer.from_pretrained(args.model_path or spec["hf_id"],
                                                        trust_remote_code=bool(spec.get("trust_remote_code"))), args.model)
    chat = spec.get("prompt_format", "plain") == "chat"
    out = Path(args.out) / args.model
    st = build_stimuli(tok, out, list(prof["families"]), list(prof["levels"]), int(prof["contexts_per_cell"]),
                       int(prof["seed"]), chat=chat, max_context=args.max_context)
    print({f"{k[0]}_{k[1]}": len(v) for k, v in st.items()}, "->", out / "stimuli")


def cmd_calibrate(args) -> None:
    args.exp, args.force, args.skip_repro, args.analyze, args.gsm8k_path = "calibrate", False, True, False, None
    cmd_run(args)


def cmd_analyze(args) -> None:
    from .analysis.report import analyze

    rep = analyze(Path(args.out) / args.model if args.model else Path(args.dir))
    print("figures:", rep["figures"])


def cmd_agreement(args) -> None:
    from .experiments.exp_c import agreement

    print(agreement(args.csv))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m loopdetail")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_args(sp, need_model: bool = True) -> None:
        sp.add_argument("--model", required=need_model, help="registry name (configs/models.yaml)")
        sp.add_argument("--model-path", default=None, help="local path / HF id overriding hf_id")
        sp.add_argument("--models-yaml", type=Path, default=None, help="extra registry YAML")
        sp.add_argument("--experiments-yaml", type=Path, default=None)
        sp.add_argument("--device", default=None)
        sp.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--out", default=str(DEFAULT_RESULTS))

    r = sub.add_parser("run", help="run experiments")
    model_args(r)
    r.add_argument("--exp", default="sanity,B,A1,A2,H3,C",
                   help="comma list; run `calibrate` first (or include it) so levels are selected")
    r.add_argument("--profile", default="full")
    r.add_argument("--smoke", action="store_true", help="tiny end-to-end check (results meaningless)")
    r.add_argument("--skip-repro", action="store_true", help="skip S2 (ARC per-loop trend)")
    r.add_argument("--force", action="store_true", help="run even if sanity did not PASS")
    r.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None)
    r.add_argument("--analyze", action="store_true", help="run `analyze` afterwards")
    r.add_argument("--gsm8k-path", default=None, help="local GSM8K dir with {train,test}.jsonl (ExpC)")
    r.set_defaults(fn=cmd_run)

    i = sub.add_parser("inspect", help="verify adapter assumptions on a model")
    model_args(i)
    i.add_argument("--loops", type=int, default=None)
    i.add_argument("--no-generation", action="store_true")
    i.add_argument("--out-json", default=None)
    i.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("stimuli", help="build stimuli for a model's tokenizer only")
    model_args(s)
    s.add_argument("--profile", default="full")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--max-context", type=int, default=None, help="skip levels longer than this")
    s.set_defaults(fn=cmd_stimuli)

    c = sub.add_parser("calibrate", help="pick difficulty levels (alias for run --exp calibrate)")
    model_args(c)
    c.add_argument("--profile", default="full")
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--chat", action=argparse.BooleanOptionalAction, default=None)
    c.set_defaults(fn=cmd_calibrate)

    a = sub.add_parser("analyze", help="statistics + figures for a results directory")
    a.add_argument("--model", default=None)
    a.add_argument("--out", default=str(DEFAULT_RESULTS))
    a.add_argument("--dir", default=None, help="explicit results/<model> directory")
    a.set_defaults(fn=cmd_analyze)

    g = sub.add_parser("agreement", help="auto vs human detail-error agreement (ExpC CSV)")
    g.add_argument("--csv", required=True)
    g.set_defaults(fn=cmd_agreement)

    args = p.parse_args(argv)
    set_threads(getattr(args, "threads", 4))
    args.fn(args)
