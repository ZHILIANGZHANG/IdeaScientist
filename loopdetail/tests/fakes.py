"""Tiny randomly-initialised looped models + a word-level tokenizer for CPU tests.

FakeOuro  : embed -> shared stack of 2 sandwich-norm decoder layers looped T times, final
            RMSNorm after EVERY loop (fed into the next loop) -> lm_head. Mirrors the
            Ouro HF module names and the OuroModel.forward return convention.
FakeRaven : wte -> prelude(1) -> [adapter(cat[state, prelude_out]) -> core_block(2)] x r
            from a random/zero initial state -> ln_f -> coda(1) -> ln_f -> lm_head, with
            fused Wqkv + `chunks` and Huginn's forward(num_steps, input_states) signature.
Outputs are meaningless; the fakes exist only to exercise adapters and pipelines.
"""
from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..stimuli.vocab import BOX_LABELS, CATEGORY_ITEMS, NAMES
from ..stimuli.render import TEMPLATES


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def _rope(x: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, D] rotate-half RoPE with positions 0..L-1."""
    L, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    ang = torch.outer(torch.arange(L, device=x.device).float(), inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int, n_kv: int,
            head_dim: int) -> torch.Tensor:
    B, L, _ = q.shape
    q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, L, n_kv, head_dim).transpose(1, 2)
    v = v.view(B, L, n_kv, head_dim).transpose(1, 2)
    q, k = _rope(q), _rope(k)
    rep = n_heads // n_kv
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.transpose(1, 2).reshape(B, L, n_heads * head_dim)


class MLP(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, h, False), nn.Linear(d, h, False), nn.Linear(h, d, False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OuroAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, d // n_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        y = _attend(self.q_proj(x), self.k_proj(x), self.v_proj(x), self.n_heads, self.n_kv, self.head_dim)
        return self.o_proj(y)


class OuroDecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int):
        super().__init__()
        self.self_attn = OuroAttention(d, n_heads, n_kv)
        self.mlp = MLP(d, 2 * d)
        self.input_layernorm, self.input_layernorm_2 = RMSNorm(d), RMSNorm(d)
        self.post_attention_layernorm, self.post_attention_layernorm_2 = RMSNorm(d), RMSNorm(d)

    def forward(self, x):
        x = x + self.input_layernorm_2(self.self_attn(self.input_layernorm(x)))
        return x + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x)))


class FakeOuroModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, d)
        self.layers = nn.ModuleList(OuroDecoderLayer(d, config.num_attention_heads, config.num_key_value_heads)
                                    for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(d)
        self.early_exit_gate = nn.Linear(d, 1)
        self.total_ut_steps = config.total_ut_steps

    def forward(self, input_ids, use_cache=False, **_):
        h = self.embed_tokens(input_ids)
        hs, gates = [], []
        for _ in range(self.total_ut_steps):
            for layer in self.layers:
                h = layer(h)
            h = self.norm(h)
            hs.append(h)
            gates.append(self.early_exit_gate(h))
        return SimpleNamespace(last_hidden_state=h), hs, gates


class FakeOuroForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = FakeOuroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, **kw):
        _, hs, _ = self.model(input_ids)
        return SimpleNamespace(logits=self.lm_head(hs[-1]))


class RavenAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int):
        super().__init__()
        self.n_head, self.n_kv_heads, self.head_dim = n_heads, n_kv, d // n_heads
        self.chunks = [d, n_kv * self.head_dim, n_kv * self.head_dim]
        self.Wqkv = nn.Linear(d, sum(self.chunks), bias=False)
        self.qk_bias = nn.Parameter(torch.randn(2, 1, n_heads, self.head_dim) * 0.02)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, L, self.n_head, self.head_dim) + self.qk_bias[0]
        k = k.view(B, L, self.n_kv_heads, self.head_dim) + self.qk_bias[1, :, : self.n_kv_heads]
        y = _attend(q.reshape(B, L, -1), k.reshape(B, L, -1), v, self.n_head, self.n_kv_heads, self.head_dim)
        return self.proj(y)


class SandwichBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, n_kv: int):
        super().__init__()
        self.norm_1, self.norm_2, self.norm_3, self.norm_4 = RMSNorm(d), RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.attn = RavenAttention(d, n_heads, n_kv)
        self.mlp = MLP(d, 2 * d)

    def forward(self, x):
        x = self.norm_2(self.attn(self.norm_1(x)) + x)
        return self.norm_4(self.mlp(self.norm_3(x)) + x)


class FakeRavenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, h, kv = config.n_embd, config.num_attention_heads, config.num_key_value_heads
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, d),
            prelude=nn.ModuleList(SandwichBlock(d, h, kv) for _ in range(config.n_layers_in_prelude)),
            adapter=nn.Linear(2 * d, d, bias=False),
            core_block=nn.ModuleList(SandwichBlock(d, h, kv) for _ in range(config.n_layers_in_recurrent_block)),
            coda=nn.ModuleList(SandwichBlock(d, h, kv) for _ in range(config.n_layers_in_coda)),
            ln_f=RMSNorm(d),
        ))
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.emb_scale = 1.0

    def initialize_state(self, input_embeds, scale: float = 1.0):
        x = torch.randn_like(input_embeds)
        std = self.config.init_values["std"] * scale
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return x * self.emb_scale

    def forward(self, input_ids, input_states=None, num_steps=None, output_details=None, **_):
        e = self.transformer.wte(input_ids) * self.emb_scale
        for block in self.transformer.prelude:
            e = block(e)
        x = self.initialize_state(e) if input_states is None else input_states.clone()
        steps = self.config.mean_recurrence if num_steps is None else int(num_steps)
        for _ in range(steps):
            x = self.transformer.adapter(torch.cat([x, e], dim=-1))
            for block in self.transformer.core_block:
                x = block(x)
        latents = x.clone()
        x = self.transformer.ln_f(x)
        for block in self.transformer.coda:
            x = block(x)
        x = self.transformer.ln_f(x)
        return SimpleNamespace(logits=self.lm_head(x).float(), latent_states=latents)


class WordTokenizer:
    """Word-level tokenizer: pieces are ' word', 'word', single digits, punctuation, newline.
    Unknown pieces fall back to characters. Implements the TextTokenizer protocol."""

    _piece = re.compile(r" ?[A-Za-z]+| ?\d|\n| ?[^\sA-Za-z\d]| ")

    def __init__(self, words: Sequence[str]):
        self.name = "fake-word-tokenizer"
        vocab = ["<bos>", "<eos>", "\n", " "] + [str(d) for d in range(10)] + [f" {d}" for d in range(10)]
        vocab += [c for c in ".,:?!'-()"] + [f" {c}" for c in ".,:?!'-()"]
        vocab += [chr(c) for c in range(97, 123)] + [chr(c) for c in range(65, 91)]
        for w in words:
            vocab += [w, " " + w]
        self.vocab = list(dict.fromkeys(vocab))
        self.index = {p: i for i, p in enumerate(self.vocab)}
        self.bos, self.eos = 0, 1

    def __len__(self) -> int:
        return len(self.vocab)

    def encode_offsets(self, text: str, add_special: bool = True):
        ids, offs = ([self.bos], [(0, 0)]) if add_special else ([], [])
        for m in self._piece.finditer(text):
            piece, s = m.group(0), m.start()
            if piece in self.index:
                ids.append(self.index[piece])
                offs.append((s, m.end()))
            else:
                for j, ch in enumerate(piece):
                    ids.append(self.index.get(ch, self.index[" "]))
                    offs.append((s + j, s + j + 1))
        return ids, offs

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        return self.encode_offsets(text, add_special)[0]

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] for i in ids if i > 1)

    def chat_prompt(self, user_text: str) -> str:
        return user_text

    @property
    def has_chat_template(self) -> bool:
        return False

    @property
    def stop_ids(self) -> list[int]:
        return [self.eos]


def fake_tokenizer() -> WordTokenizer:
    words = [w for items in CATEGORY_ITEMS.values() for w in items] + list(CATEGORY_ITEMS) + NAMES
    words += BOX_LABELS[:200] + [f"{d}000s" for d in range(1, 10)]
    for tpl in TEMPLATES.values():
        for s in tpl.values():
            words += re.findall(r"[A-Za-z]+", s)
    words += ["Question", "Answer", "The", "answer", "is", "step", "Let", "think", "by"]
    return WordTokenizer(sorted(set(words)))


def build_fake_ouro(seed: int = 0, total_ut_steps: int = 4):
    torch.manual_seed(seed)
    tok = fake_tokenizer()
    cfg = SimpleNamespace(vocab_size=len(tok), hidden_size=64, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, num_hidden_layers=2,
                          total_ut_steps=total_ut_steps)
    return FakeOuroForCausalLM(cfg).eval(), tok


def build_fake_huginn(seed: int = 0):
    torch.manual_seed(seed)
    tok = fake_tokenizer()
    cfg = SimpleNamespace(vocab_size=len(tok), n_embd=64, num_attention_heads=4, num_key_value_heads=4,
                          n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=1,
                          mean_recurrence=8, init_values={"std": math.sqrt(2 / (5 * 64))},
                          test_time_noise=0)
    return FakeRavenForCausalLM(cfg).eval(), tok
