"""Deep-thin gated Llama trunk + name/retrieve heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # d=512/22L/ffn2.5 from the handoff lands at ~70M; this shape is ~44M.
    vocab_size: int = 8192
    d_model: int = 448
    n_layers: int = 20
    n_heads: int = 8
    n_kv: int = 4
    ffn_mult: float = 2.0
    rope_theta: float = 10000.0
    max_seq: int = 512
    retrieve_k: int = 5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        n = int(self.d_model * self.ffn_mult)
        return (n + 7) // 8 * 8


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 only for the reduction; the (b,t,1) rsqrt is tiny, so backward
        # retains x in its native dtype instead of a full fp32 copy
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * rms * self.weight


def rotate(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attn(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h, kv, d = cfg.n_heads, cfg.n_kv, cfg.head_dim
        self.n_heads, self.n_kv, self.head_dim = h, kv, d
        self.q = nn.Linear(cfg.d_model, h * d, bias=False)
        self.k = nn.Linear(cfg.d_model, kv * d, bias=False)
        self.v = nn.Linear(cfg.d_model, kv * d, bias=False)
        self.o = nn.Linear(h * d, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, h * d, bias=False)
        self.q_norm = RMSNorm(d)
        self.k_norm = RMSNorm(d)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q(x).view(b, t, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(b, t, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        attn_mask = None
        causal = True
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            past = k.shape[2] - t
            if past > 0:
                causal = False
                if t > 1:  # multi-token feed onto existing cache: offset causal mask
                    i = torch.arange(t, device=x.device)[:, None]
                    j = torch.arange(k.shape[2], device=x.device)[None, :]
                    attn_mask = (j <= past + i)[None, None]
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        y = y * torch.sigmoid(self.gate(x))
        return self.o(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attn(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.n3 = RMSNorm(cfg.d_model)
        self.n4 = RMSNorm(cfg.d_model)
        hidden = cfg.ffn_dim
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None
    ) -> torch.Tensor:
        x = x + self.n2(self.attn(self.n1(x), cos, sin, cache))
        n = self.n3(x)
        x = x + self.n4(self.w2(F.silu(self.w1(n)) * self.w3(n)))
        return x


class ToolTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.name_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer("inv_freq", _inv_freq(cfg), persistent=False)

    def rope(self, t: int, device: torch.device, dtype: torch.dtype, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(offset, offset + t, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None, :, :].to(dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None, :, :].to(dtype)
        return cos, sin

    def forward(
        self, ids: torch.Tensor, caches: list[dict] | None = None, need_logits: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """caches: per-layer dicts mutated in place for incremental decoding;
        RoPE offset is inferred from the cached length.

        need_logits=False skips the vocabulary projection — on grammar-forced
        tokens the output is already determined, so only the KV cache update
        matters and the |vocab| matmul is pure waste.
        """
        x = self.embed(ids)
        offset = caches[0]["k"].shape[2] if caches and "k" in caches[0] else 0
        cos, sin = self.rope(ids.shape[1], ids.device, x.dtype, offset=offset)
        for li, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, caches[li] if caches is not None else None)
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight) if need_logits else None
        return logits, x

    def new_caches(self) -> list[dict]:
        return [{} for _ in range(self.cfg.n_layers)]

    def name_scores(
        self, hidden: torch.Tensor, decision_pos: int, cand_spans: list[tuple[int, int]], batch: int = 0
    ) -> torch.Tensor:
        """Readout head: score each candidate tool for the call at decision_pos.

        hidden: (B, T, D) from forward(). cand_spans are token [start, end) spans of each
        candidate's name inside the prompt. Bilinear h_dec . W . mean(h_span).
        """
        h = hidden[batch]
        dec = self.name_head(h[decision_pos])
        segs = torch.stack([h[s:e].mean(dim=0) for s, e in cand_spans])
        return segs @ dec

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _inv_freq(cfg: Config) -> torch.Tensor:
    d = cfg.head_dim
    return 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2).float() / d))


def build(vocab_size: int, **kw) -> ToolTransformer:
    return ToolTransformer(Config(vocab_size=vocab_size, **kw))
