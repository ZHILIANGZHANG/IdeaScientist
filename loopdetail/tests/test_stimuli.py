from collections import Counter

import pytest

from loopdetail.experiments.calibrate import select
from loopdetail.stimuli.build import build_stimuli, check_instance
from loopdetail.stimuli.generate import FAMILIES, generate, variant_values
from loopdetail.stimuli.pools import is_single_token, resolve_pool
from loopdetail.stimuli.render import render, render_context_only, render_knowledge
from loopdetail.stimuli.vocab import CATEGORY_ITEMS, FILLERS, NAMES
from loopdetail.tests.fakes import WordTokenizer, fake_tokenizer

L8 = {"name": "N8", "N": 8}


@pytest.fixture(scope="module")
def tok():
    return fake_tokenizer()


@pytest.fixture(scope="module")
def pool(tok):
    return resolve_pool(tok)


def test_single_token_filtering_drops_multi_token_words():
    words = [w for c, items in CATEGORY_ITEMS.items() for w in items if w not in ("apple", "hammer")]
    words += list(CATEGORY_ITEMS) + ["Alice", "Bob"]
    tok = WordTokenizer(words)
    assert not is_single_token(tok, "apple") and is_single_token(tok, "banana")
    p = resolve_pool(tok, min_names=2)
    assert "apple" in p.dropped["items"] and "hammer" in p.dropped["items"]
    all_items = [i for items in p.categories.values() for i in items]
    assert "apple" not in all_items
    assert all(is_single_token(tok, w) for w in all_items + list(p.categories))
    assert len({len(v) for v in p.categories.values()}) == 1  # equal-size categories


def test_fillers_are_neutral():
    banned = {w for items in CATEGORY_ITEMS.values() for w in items} | set(CATEGORY_ITEMS) | set(NAMES)
    for f in FILLERS:
        assert not any(ch.isdigit() for ch in f)
        assert not {w.strip(".,").lower() for w in f.split()} & {b.lower() for b in banned}, f


@pytest.mark.parametrize("family", FAMILIES)
def test_balanced_design(pool, family):
    insts = generate(family, pool, L8, 256, seed=3)
    for key in ("detail", "gist"):
        slots = Counter(i[key]["answer"] for i in insts)
        assert set(slots) == {0, 1, 2, 3} and max(slots.values()) - min(slots.values()) <= 4
    quart = Counter(i["query_quartile"] for i in insts)
    assert max(quart.values()) - min(quart.values()) <= 4
    for i in insts:
        q = i["query_idx"]
        assert q // 2 == i["query_quartile"] and i["distance"] == 7 - q
        assert len(set(i["labels"])) == 8 and len(set(i["values"])) == 8
        assert i["detail"]["candidates"][i["detail"]["answer"]] == i["values"][q]
        assert len(set(i["detail"]["candidates"])) == 4 and len(set(i["gist"]["candidates"])) == 4


def test_explicit_and_implicit_are_paired(pool):
    a = generate("box_explicit", pool, L8, 20, seed=1)
    b = generate("box", pool, L8, 20, seed=1)
    for x, y in zip(a, b):
        assert x["values"] == y["values"] and x["labels"] == y["labels"] and x["detail"] == y["detail"]


def test_far_quartile_and_large_n(pool):
    far = generate("box_explicit", pool, {"name": "far", "N": 64, "quartiles": [0]}, 30, seed=2)
    assert all(i["query_quartile"] == 0 and i["query_idx"] < 16 for i in far)
    big = generate("box", pool, {"name": "N128", "N": 128}, 5, seed=2)
    for i in big:  # values may repeat, but the queried value is unique in its context
        assert len(i["values"]) == 128 and i["values"].count(i["values"][i["query_idx"]]) == 1
    codes = generate("codes", pool, {"name": "N128", "N": 128}, 3, seed=2)
    assert all(len(set(i["labels"])) == 128 and len(set(i["values"])) == 128 for i in codes)


def test_answer_frequency_balanced(pool):
    insts = generate("box", pool, {"name": "N4", "N": 4}, 10 * sum(len(v) for v in pool.categories.values()), seed=1)
    freq = Counter(i["values"][i["query_idx"]] for i in insts)
    assert max(freq.values()) - min(freq.values()) <= 1


def test_box_foils_and_minimal_pairs(pool):
    cat_of = pool.item_category()
    for inst in generate("box", pool, {"name": "N16", "N": 16}, 50, seed=5):
        q = inst["query_idx"]
        true = inst["values"][q]
        assert all(cat_of[c] == cat_of[true] for c in inst["detail"]["candidates"])
        assert inst["gist"]["candidates"][inst["gist"]["answer"]] == cat_of[true]
        d, g, u = (variant_values(inst, v) for v in ("detail_swap", "gist_swap", "unrelated_swap"))
        assert sum(a != b for a, b in zip(d, inst["values"])) == 1 and cat_of[d[q]] == cat_of[true]
        assert sum(a != b for a, b in zip(g, inst["values"])) == 1 and cat_of[g[q]] != cat_of[true]
        j = inst["variants"]["unrelated_swap"]["idx"]
        assert j != q and u[q] == true and cat_of[u[j]] != cat_of[inst["values"][j]]
        ctrl = inst["control_values"]
        assert all(a != b for a, b in zip(ctrl, inst["values"])) and cat_of[ctrl[q]] != cat_of[true]
        assert all(v in inst["category_of"] for v in d + g + u + ctrl)


def test_codes_foils_and_minimal_pairs(pool):
    for inst in generate("codes", pool, L8, 50, seed=5):
        q = inst["query_idx"]
        true = inst["values"][q]
        for c in inst["detail"]["candidates"]:
            assert c[0] == true[0] and sum(a != b for a, b in zip(c, true)) in (0, 1)
        assert inst["gist"]["candidates"][inst["gist"]["answer"]] == f"{true[0]}000s"
        assert variant_values(inst, "detail_swap")[q][0] == true[0]
        assert variant_values(inst, "gist_swap")[q][0] != true[0]
        assert inst["control_values"][q][0] != true[0]


@pytest.mark.parametrize("family", FAMILIES)
def test_render_spans_and_tokenisation(tok, pool, family):
    level = {"name": "N8_f2", "N": 8, "fillers": 2}
    for inst in generate(family, pool, level, 10, seed=2):
        check_instance(tok, inst, chat=False)
        pr = render(inst, "detail")
        for (s, e), v in zip(pr.fact_spans, inst["values"]):
            assert pr.text[s:e] == v
        assert sum(pr.text.count(f) for f in FILLERS) >= 8  # 2 fillers after each fact (may repeat)
        if family.endswith("explicit"):
            gold = inst["gist"]["candidates"][inst["gist"]["answer"]]
            assert pr.text[slice(*pr.gist_spans[inst["query_idx"]])] == gold
        else:
            assert pr.gist_spans is None
        nc = render(inst, "gist", with_context=False)
        assert nc.fact_spans == [] and inst["values"][0] not in nc.text.split("Question")[0]
        co = render_context_only(inst)
        assert all(co.text[s:e] == v for (s, e), v in zip(co.fact_spans, inst["values"]))
        kn = render_knowledge(inst)
        assert inst["values"][inst["query_idx"]] in kn.text and kn.candidates == nc.candidates
        assert kn.answer == inst["gist"]["answer"] and "Box" not in kn.text


@pytest.mark.parametrize("family", ["box_explicit", "box"])
def test_minimal_pair_token_length_preserved(tok, pool, family):
    for inst in generate(family, pool, {"name": "N8_f1", "N": 8, "fillers": 1}, 10, seed=4):
        lens = {len(tok.encode(render(inst, "detail", values=variant_values(inst, v)).text))
                for v in ("base", "detail_swap", "gist_swap", "unrelated_swap", "control")}
        assert len(lens) == 1


def test_build_reload_and_context_skip(tok, tmp_path):
    levels = [{"name": "N4", "N": 4}, {"name": "N64_f2", "N": 64, "fillers": 2}]
    a = build_stimuli(tok, tmp_path, ["box_explicit", "codes"], levels, 6, seed=9)
    b = build_stimuli(tok, tmp_path, ["box_explicit", "codes"], levels, 6, seed=9)
    assert a[("box_explicit", "N4")] == b[("box_explicit", "N4")] and len(a) == 4
    assert (tmp_path / "stimuli" / "pool.json").exists()
    c = build_stimuli(tok, tmp_path / "small", ["box_explicit"], levels, 6, seed=9, max_context=400)
    assert ("box_explicit", "N4") in c and ("box_explicit", "N64_f2") not in c


def test_calibration_selection():
    rows = [{"level": "N4", "detail_acc": 1.0}, {"level": "N32", "detail_acc": 0.95},
            {"level": "N64", "detail_acc": 0.7}, {"level": "N128", "detail_acc": 0.45}]
    order = ["N4", "N32", "N64", "N128"]
    assert select(rows, order, 0.4, 0.9)[0] == ["N4", "N64", "N128"]
    sel, note = select([{"level": "N4", "detail_acc": 1.0}, {"level": "N32", "detail_acc": 0.97}], order, 0.4, 0.9)
    assert sel == ["N4", "N32"] and "closest" in note
    ceiling = [{"level": lv, "detail_acc": 1.0} for lv in order]  # all at ceiling -> hardest level
    assert select(ceiling, order, 0.4, 0.9)[0] == ["N4", "N128"]


def test_label_modes(pool, tok):
    num = generate("box_explicit", pool, {"name": "n", "N": 32, "labels": "confusable"}, 5, seed=1)
    assert all(lab.isdigit() and len(lab) == 3 for i in num for lab in i["labels"])
    conf = generate("codes_explicit", pool, {"name": "c", "N": 32, "labels": "confusable"}, 5, seed=1)
    for i in conf:
        assert len({lab.split()[0] for lab in i["labels"]}) <= 4 and len(set(i["labels"])) == 32
        check_instance(tok, i, chat=False)
    for i in num:
        check_instance(tok, i, chat=False)
