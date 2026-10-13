"""End-to-end smoke runs of every experiment on the fake looped models (pipeline only;
the numbers are meaningless)."""
import json

import pytest

from loopdetail.cli import main
from loopdetail.utils import read_json


@pytest.fixture(scope="module")
def gsm8k_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gsm8k")
    rows = [{"question": f"Tom has {i} apples and buys 3 more. How many apples?",
             "answer": f"{i} + 3 = <<{i}+3={i + 3}>>{i + 3}\n#### {i + 3}"} for i in range(1, 9)]
    for split in ("train", "test"):
        (d / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


@pytest.mark.parametrize("model", ["fake-ouro", "fake-huginn"])
def test_smoke_all_experiments(model, tmp_path, gsm8k_dir):
    main(["run", "--model", model, "--smoke", "--skip-repro", "--out", str(tmp_path),
          "--exp", "calibrate,sanity,B,A1,A2,H3,C", "--gsm8k-path", str(gsm8k_dir), "--analyze"])
    root = tmp_path / "smoke" / model
    cal = read_json(root / "calibration.json")
    assert set(cal["selected"]) == {"box_explicit", "codes_explicit", "box"}
    assert cal["selected"]["box"] == cal["selected"]["box_explicit"] and cal["selected"]["box"][0] == "N4"
    rep = read_json(root / "sanity" / "sanity_report.json")
    assert rep["checks"]["S1_readout_equivalence"]["status"] == "PASS"
    assert rep["checks"]["S3_patch_noop_fp32"]["status"] == "PASS"
    assert rep["checks"]["S4_probe_current_token"]["status"] == "PASS"
    for exp in ("B", "A1", "A2", "H3", "C"):
        s = read_json(root / exp / "summary.json")
        assert s["meta"]["git_hash"] and s["meta"]["timestamp_utc"]
    b = read_json(root / "B" / "summary.json")
    assert {r["t"] for r in b["table"]} == set(b["loops"])
    assert {r["level"] for r in b["table"]} <= set(cal["selected"]["box_explicit"])
    assert "table_category_known" in b and all("cat_known_rate" in r for r in b["table"])
    a1 = [json.loads(line) for line in (root / "A1" / "rows" / "box_explicit.jsonl").read_text().splitlines()]
    assert any(r["what"] == "embed" for r in a1) and all("mdl" in r for r in a1)
    assert all("acc_minus_l1l0" in r and "acc_minus_embed" in r for r in a1 if r["what"] != "embed")
    h3 = read_json(root / "H3" / "summary.json")
    conds = {r["condition"] for r in h3["table"]}
    assert conds == {"none", "same_loop1", "diff_loop1", "noise"}
    assert (root / "C" / "annotation_sample.csv").exists()
    assert (root / "analysis.json").exists()
    # resumability: a second run skips finished shards and reproduces the summaries
    main(["run", "--model", model, "--smoke", "--out", str(tmp_path), "--exp", "B",
          "--gsm8k-path", str(gsm8k_dir)])
    assert read_json(root / "B" / "summary.json")["table"] == b["table"]
