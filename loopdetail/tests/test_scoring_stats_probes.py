import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats as sps

from loopdetail import gsm8k
from loopdetail.analysis.probes import control_labels, group_folds, online_code_mdl, probe_cell
from loopdetail.analysis.stats import (adjacent_mcnemar, bootstrap_ci, loop_info_interaction,
                                       mcnemar_exact, wilson_ci)
from loopdetail.experiments.exp_c import agreement, summarize_c
from loopdetail.scoring import candidate_token_ids, choice_stats, score_loops, score_with


def test_multi_token_scoring_matches_manual(fake_ouro):
    tok = fake_ouro.tok
    prompt = tok.encode("The code of Alice is")
    cands = candidate_token_ids(tok, [" 4821", " 4831"])
    assert all(len(c) == 4 for c in cands)
    lp = score_loops(fake_ouro, prompt, cands, [1, 3], "set")
    ids = torch.tensor([prompt + cands[0]])
    logits = fake_ouro.readout_logits(ids, 3, positions=list(range(ids.shape[1])))
    logp = torch.log_softmax(logits, -1)
    manual = sum(float(logp[len(prompt) - 1 + j, c]) for j, c in enumerate(cands[0]))
    assert lp[3][0] == pytest.approx(manual, abs=1e-4)
    cap = score_loops(fake_ouro, prompt, cands, [1, 3], "capture")
    assert np.allclose(cap[1], lp[1], atol=1e-4) and np.allclose(cap[3], lp[3], atol=1e-4)


def test_single_token_scoring_one_forward(fake_huginn):
    tok = fake_huginn.tok
    prompt = tok.encode("Box AB contains the")
    cands = candidate_token_ids(tok, [" apple", " pear", " plum", " fig"])
    lp = score_with(lambda ids, pos: fake_huginn.readout_logits(ids, 2, pos), prompt, cands, torch.device("cpu"))
    full = torch.log_softmax(fake_huginn.readout_logits(torch.tensor([prompt]), 2)[0], -1)
    assert np.allclose(lp, [float(full[c[0]]) for c in cands], atol=1e-5)


def test_choice_stats():
    st = choice_stats(np.array([-1.0, -0.5, -3.0, -2.0]), 1, np.array([-1.0, -2.0, -3.0, -0.1]))
    assert st["correct"] == 1 and st["margin"] == pytest.approx(0.5)
    assert st["correct_cal"] == 1 and st["margin_cal"] == pytest.approx(1.5 - 0.0)


def test_wilson_and_mcnemar():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.192, abs=0.01)
    a = [1] * 10 + [0] * 3 + [1] * 20
    b = [0] * 10 + [1] * 3 + [1] * 20
    r = mcnemar_exact(a, b)
    assert r["a_only"] == 10 and r["b_only"] == 3
    assert r["p"] == pytest.approx(sps.binomtest(3, 13, 0.5).pvalue, rel=1e-9)


def test_bootstrap_ci_clusters():
    vals = np.r_[np.ones(50), np.zeros(50)]
    lo, hi = bootstrap_ci(vals, np.repeat(np.arange(10), 10), n_boot=500, seed=0)
    assert lo < 0.5 < hi


def test_interaction_recovers_sign():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(150):
        u = rng.normal(0, 0.5)
        for t in (1, 2, 4, 8):
            lt = np.log2(t)
            for info, slope in (("detail", -0.6), ("gist", 0.2)):
                p = 1 / (1 + np.exp(-(1.0 + u + slope * lt)))
                rows.append({"context_id": f"c{c}", "item_id": f"c{c}", "t": t, "lt": lt - 1.5,
                             "info_type": info, "correct": int(rng.random() < p), "family": "box", "N": 8})
    df = pd.DataFrame(rows)
    res = loop_info_interaction(df)
    assert res["gee"]["coef"] < 0 and res["gee"]["ci95"][1] < 0
    assert res["mixed_vb"]["coef"] < 0
    mc = adjacent_mcnemar(df)
    assert len(mc) == 2 * 3


def test_probe_on_synthetic_features():
    rng = np.random.default_rng(0)
    n, d, k = 600, 20, 4
    y = rng.integers(0, k, n)
    centers = rng.normal(0, 3, (k, d))
    X = centers[y] + rng.normal(0, 1, (n, d))
    groups = np.repeat(np.arange(n // 4), 4)
    test = groups >= 120
    types = np.array([f"w{i % 40}" for i in range(n)])
    res = probe_cell(X, y.astype(str), types, groups, test, [0.1, 1.0], 3, 0, n_boot=100)
    assert res["acc"] > 0.95 and res["acc_ci95"][0] > 0.9
    noise = probe_cell(rng.normal(size=(n, d)), y.astype(str), types, groups, test, [0.1, 1.0], 3, 0, n_boot=50)
    assert noise["acc"] < 0.45


def test_control_labels_and_folds():
    types = np.array(["a", "b", "a", "c", "b"])
    lab = control_labels(types, 3, 0)
    assert lab[0] == lab[2] and lab[1] == lab[4]
    folds = group_folds(np.array([0, 0, 1, 1, 2, 2, 3]), 2, 0)
    seen = np.concatenate(folds)
    assert sorted(seen) == list(range(7))
    g = np.array([0, 0, 1, 1, 2, 2, 3])
    assert all(len(set(g[f]) & set(g[h])) == 0 for i, f in enumerate(folds) for h in folds[i + 1:])


def test_online_mdl_compresses_signal():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    X = np.eye(3)[y] * 3 + rng.normal(0, 0.5, (400, 3))
    assert online_code_mdl(X, y, 3, 1.0, 0)["compression"] > 1.5


def test_gsm8k_tagging_and_extraction():
    q = "Tom has 3 apples and buys twelve more. Each costs $2."
    g = "He has 3 + 12 = 15 apples. They cost 15 * 2 = 30 dollars. Tax: 30 + 7 = 37.\n#### 37"
    tags = gsm8k.tag_operands(q, g)
    srcs = [o["source"] for e in tags["equations"] for o in e["operands"]]
    assert srcs == ["problem", "problem", "computed", "problem", "computed", "constant"]
    assert tags["detail_error"] == 0
    assert gsm8k.tag_operands(q, "So 3 * 17 = 51")["unexplained_operands"] == [17.0]
    assert gsm8k.extract_answer(g) == 37.0
    assert gsm8k.extract_answer("the answer is 1,250.\nQuestion: next") == 1250.0
    assert gsm8k.gold_answer("blah\n#### 1,000") == 1000.0


def test_exp_c_summary_and_agreement(tmp_path):
    rows = [{"item_id": "a", "t": 1, "correct": 1, "detail_error": 0},
            {"item_id": "a", "t": 2, "correct": 0, "detail_error": 1},
            {"item_id": "b", "t": 1, "correct": 0, "detail_error": 1},
            {"item_id": "b", "t": 2, "correct": 1, "detail_error": 0}]
    s = summarize_c(rows)
    assert s["regression_set"][2] == ["a"] and s["per_t"][1]["acc"] == 0.5
    p = tmp_path / "ann.csv"
    p.write_text("item_id,t,question,generation,pred,gold,correct,auto_detail_error,auto_unexplained_operands,"
                 "human_detail_error,human_notes\n"
                 "a,1,q,g,1,1,1,1,,1,\nb,1,q,g,1,1,1,0,,0,\nc,1,q,g,1,1,1,1,,0,\nd,1,q,g,1,1,1,0,,,\n")
    r = agreement(str(p))
    assert r["n"] == 3 and r["percent_agreement"] == pytest.approx(2 / 3)
