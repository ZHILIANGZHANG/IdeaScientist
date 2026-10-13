import pytest
import torch

from loopdetail.models.base import (AdapterAssumptionError, KVPatch, ModulePaths, noise_patch_like,
                                    patch_from_capture)
from loopdetail.models.ouro import OuroAdapter
from loopdetail.models.raven import RavenAdapter
from loopdetail.tests.fakes import build_fake_huginn, build_fake_ouro


def test_kv_projection_called_once_per_loop(looped, ids):
    calls = {}
    hooks = [s.module.register_forward_hook(lambda m, i, o, k=si: calls.__setitem__(k, calls.get(k, 0) + 1))
             for si, s in enumerate(looped.kv_sites)]
    try:
        for T in (1, 3, 5):
            calls.clear()
            looped._run(ids, T)
            assert set(calls.values()) == {T}
            assert len(calls) == len(looped.kv_sites)
    finally:
        for h in hooks:
            h.remove()


def test_prelude_and_coda_run_once(fake_huginn):
    ids = torch.tensor([fake_huginn.tok.encode("Box AB contains the apple.")])
    calls = []
    hooks = [m.register_forward_hook(lambda *a: calls.append(1)) for m in fake_huginn.prelude + fake_huginn.coda]
    try:
        fake_huginn._run(ids, 6)
    finally:
        for h in hooks:
            h.remove()
    assert len(calls) == 2


def test_capture_shapes(looped, ids):
    pos = [2, 4, ids.shape[1] - 1]
    cap = looped.capture(ids, 4, pos, loops=[1, 2, 4], layers=[0, 1])
    assert cap.k.shape == (3, 2, 3, looped.n_kv_heads, looped.head_dim)
    assert cap.v.shape == cap.k.shape
    assert cap.hidden.shape == (3, 2, 3, looped.d_model)
    assert cap.k.dtype == torch.float16


def test_capture_values_differ_across_loops(looped, ids):
    cap = looped.capture(ids, 3, [3], what=("v",), store_dtype=None)
    assert not torch.allclose(cap.v[0], cap.v[2])


def test_readout_equivalence(looped, ids):
    loops = [1, 2, 3, 6]
    cap = looped.readout_loops(ids, loops, positions=[1, ids.shape[1] - 1])
    for t in loops:
        ref = looped.readout_logits(ids, t, positions=[1, ids.shape[1] - 1])
        assert torch.allclose(ref, cap[t], atol=1e-5), t


def test_loops_change_output(looped, ids):
    assert not torch.allclose(looped.readout_logits(ids, 1), looped.readout_logits(ids, 4))


def test_patch_noop_exact_fp32(looped, ids):
    pos = list(range(1, ids.shape[1] - 1))
    T = 4
    cap = looped.capture(ids, T, pos, what=("k", "v"), store_dtype=None)
    vals = {(t, l, k): cap.get(k)[t - 1, l] for t in range(1, T + 1) for l in range(looped.n_layers) for k in "kv"}
    base = looped.readout_logits(ids, T)
    assert torch.equal(looped.patch_kv(ids, T, KVPatch(pos, vals)), base)


def test_patch_with_other_input_changes_logits(looped):
    a = torch.tensor([looped.tok.encode("Box AB contains the apple. Box CD contains the hammer. Box AB")])
    b = torch.tensor([looped.tok.encode("Box AB contains the tiger. Box CD contains the violin. Box AB")])
    assert a.shape == b.shape
    pos = list(range(1, a.shape[1] - 3))
    src = looped.capture(b, 1, pos, what=("k", "v"), store_dtype=None)
    patch = patch_from_capture(src, 1, [2, 3], range(looped.n_layers))
    base = looped.readout_logits(a, 3)
    out = looped.patch_kv(a, 3, patch)
    assert (out - base).abs().max() > 1e-3


def test_patch_loop1_same_input_at_loop1_is_noop(looped, ids):
    pos = list(range(1, ids.shape[1] - 1))
    src = looped.capture(ids, 1, pos, what=("k", "v"), store_dtype=None)
    patch = patch_from_capture(src, 1, [1], range(looped.n_layers))
    assert torch.equal(looped.patch_kv(ids, 3, patch), looped.readout_logits(ids, 3))


def test_head_patch_only_touches_selected_heads(looped, ids):
    pos = list(range(1, ids.shape[1] - 1))
    T = 3
    cap = looped.capture(ids, T, pos, what=("k", "v"), store_dtype=None)
    full = patch_from_capture(cap, 1, [2, 3], [0])
    head = patch_from_capture(cap, 1, [2, 3], [0], heads=[0])
    base = looped.readout_logits(ids, T)
    a, b = looped.patch_kv(ids, T, full), looped.patch_kv(ids, T, head)
    assert not torch.equal(a, base) and not torch.equal(b, base) and not torch.equal(a, b)
    # replacing only head 0 with its own values is a no-op
    own = KVPatch(pos, {(2, 0, "k"): cap.k[1, 0]}, heads=[0])
    assert torch.equal(looped.patch_kv(ids, T, own), base)


def test_noise_patch_matches_norm(looped, ids):
    pos = [1, 2, 3]
    cap = looped.capture(ids, 2, pos, what=("k", "v"), store_dtype=None)
    patch = noise_patch_like(cap, [2], [0], torch.Generator().manual_seed(0))
    new = patch.values[(2, 0, "v")]
    assert torch.allclose(new.norm(dim=-1), cap.v[1, 0].norm(dim=-1), rtol=1e-4)
    assert not torch.allclose(new, cap.v[1, 0])


def test_patch_out_of_range_loop_raises(looped, ids):
    cap = looped.capture(ids, 1, [1], what=("k",), store_dtype=None)
    patch = patch_from_capture(cap, 1, [7], [0], kinds=("k",))
    with pytest.raises(AdapterAssumptionError):
        looped.patch_kv(ids, 3, patch)


def test_bad_module_path_gives_clear_error():
    model, tok = build_fake_ouro()
    with pytest.raises(AdapterAssumptionError, match="module_paths.recurrent_layers"):
        OuroAdapter(model, tok, name="x", paths=ModulePaths(recurrent_layers="model.blocks",
                    k_proj="self_attn.k_proj", v_proj="self_attn.v_proj"), n_loops_trained=4, device="cpu")


def test_loop_state_miscount_detected():
    model, tok = build_fake_ouro()
    # lm_head is not called inside the loop -> the loop-state assumption must fail loudly
    ad = OuroAdapter(model, tok, name="x", paths=ModulePaths(recurrent_layers="model.layers",
                     k_proj="self_attn.k_proj", v_proj="self_attn.v_proj", loop_state="lm_head"),
                     n_loops_trained=4, device="cpu")
    with pytest.raises(AdapterAssumptionError, match="loop_state"):
        ad.readout_loops(torch.tensor([tok.encode("Box AB contains the apple.")]), [1, 2])


def test_raven_state_is_seeded_and_zero_option():
    model, tok = build_fake_huginn()
    paths = RavenAdapter.default_paths
    a = RavenAdapter(model, tok, name="h", paths=paths, n_loops_trained=8, device="cpu")
    assert torch.equal(a.initial_state(7), a.initial_state(7))
    z = RavenAdapter(model, tok, name="h", paths=paths, n_loops_trained=8, device="cpu",
                     options={"state_init": "zero"})
    assert z.initial_state(5).abs().sum() == 0


def test_raven_rejects_test_time_noise():
    model, tok = build_fake_huginn()
    model.config.test_time_noise = 0.1
    with pytest.raises(AdapterAssumptionError, match="test_time_noise"):
        RavenAdapter(model, tok, name="h", paths=RavenAdapter.default_paths, n_loops_trained=8, device="cpu")


def test_generate_fallback_runs(looped, ids):
    out = looped.generate_greedy(ids, 2, 5, looped.tok.stop_ids)
    assert 1 <= len(out) <= 5
