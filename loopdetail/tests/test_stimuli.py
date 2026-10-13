from collections import Counter

import numpy as np
import pytest

from loopdetail.stimuli.build import build_stimuli, check_instance
from loopdetail.stimuli.generate import generate, variant_values
from loopdetail.stimuli.pools import is_single_token, resolve_pool
from loopdetail.stimuli.render import render, render_context_only
from loopdetail.tests.fakes import WordTokenizer, fake_tokenizer
from loopdetail.stimuli.vocab import CATEGORY_ITEMS


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


@pytest.mark.parametrize("family", ["box", "codes"])
def test_balanced_design(pool, family):
    insts = generate(family, pool, 8, 256, seed=3)
    for key in ("detail", "gist"):
        slots = Counter(i[key]["answer"] for i in insts)
        assert set(slots) == {0, 1, 2, 3} and max(slots.values()) - min(slots.values()) <= 4
    quart = Counter(i["query_quartile"] for i in insts)
    assert max(quart.values()) - min(quart.values()) <= 4
    for i in insts:
        q = i["query_idx"]
        assert q // 2 == i["query_quartile"]  # N=8 -> 2 positions per quartile
        assert i["distance"] == 7 - q
        assert len(set(i["labels"])) == 8 and len(set(i["values"])) == 8
        assert i["detail"]["candidates"][i["detail"]["answer"]] == i["values"][q]
        assert len(set(i["detail"]["candidates"])) == 4 and len(set(i["gist"]["candidates"])) == 4


def test_answer_frequency_balanced(pool):
    insts = generate("box", pool, 4, 10 * sum(len(v) for v in pool.categories.values()), seed=1)
    freq = Counter(i["values"][i["query_idx"]] for i in insts)
    assert max(freq.values()) - min(freq.values()) <= 1


def test_box_foils_and_minimal_pairs(pool):
    cat_of = pool.item_category()
    for inst in generate("box", pool, 16, 50, seed=5):
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


def test_codes_foils_and_minimal_pairs(pool):
    for inst in generate("codes", pool, 8, 50, seed=5):
        q = inst["query_idx"]
        true = inst["values"][q]
        for c in inst["detail"]["candidates"]:
            assert c[0] == true[0] and sum(a != b for a, b in zip(c, true)) in (0, 1)
        assert inst["gist"]["candidates"][inst["gist"]["answer"]] == f"{true[0]}000s"
        assert variant_values(inst, "detail_swap")[q][0] == true[0]
        assert variant_values(inst, "gist_swap")[q][0] != true[0]
        assert inst["control_values"][q][0] != true[0]


def test_render_spans_and_tokenisation(tok, pool):
    for fam in ("box", "codes"):
        for inst in generate(fam, pool, 8, 10, seed=2):
            check_instance(tok, inst, chat=False)
            pr = render(inst, "detail")
            for (s, e), v in zip(pr.fact_spans, inst["values"]):
                assert pr.text[s:e] == v
            nc = render(inst, "gist", with_context=False)
            assert nc.fact_spans == [] and inst["values"][0] not in nc.text.split("Question")[0]
            co = render_context_only(inst)
            assert all(co.text[s:e] == v for (s, e), v in zip(co.fact_spans, inst["values"]))


def test_minimal_pair_token_length_preserved(tok, pool):
    for inst in generate("box", pool, 8, 10, seed=4):
        lens = {len(tok.encode(render(inst, "detail", values=variant_values(inst, v)).text))
                for v in ("base", "detail_swap", "gist_swap", "unrelated_swap", "control")}
        assert len(lens) == 1


def test_build_and_reload(tok, tmp_path):
    a = build_stimuli(tok, tmp_path, ["box", "codes"], [4], 6, seed=9)
    b = build_stimuli(tok, tmp_path, ["box", "codes"], [4], 6, seed=9)
    assert a[("box", 4)] == b[("box", 4)]
    assert (tmp_path / "stimuli" / "pool.json").exists()
    c = build_stimuli(tok, tmp_path / "other", ["box"], [4], 6, seed=10)
    assert c[("box", 4)] != a[("box", 4)]
    assert np.all([i["N"] == 4 for i in a[("codes", 4)]])
