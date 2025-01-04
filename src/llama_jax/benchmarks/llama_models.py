# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from llama_jax.checkpoint import ModelConfig

__all__ = ["Transformer"]


# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.

# ASY: Hardcode this here
max_batch_size = 5
max_seq_len = 512


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, device, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, device, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def load_parameters(config: ModelConfig, device):
    """Load Llama3 parameters from checkpoint."""
    # Validate
    input_path = config.checkpoint_path / "consolidated.00.pth"
    if not input_path.is_file():
        raise ValueError(
            f"Checkpoint {config.checkpoint_path.name} has not been converted to JAX format. See llama-jax CLI."
        )

    # Load state from checkpoint
    params = torch.load(
        input_path,
        weights_only=True,
        map_location=device,
    )

    return params


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super().__init__()
        self.config = config
        self.n_rep = config.n_heads // config.n_kv_heads

        self.wq = nn.Linear(
            config.d_model,
            config.n_heads * config.d_head,
            bias=False,
            device=device,
        )
        self.wk = nn.Linear(
            config.d_model,
            config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )
        self.wv = nn.Linear(
            config.d_model,
            config.n_kv_heads * config.d_head,
            bias=False,
            device=device,
        )
        self.wo = nn.Linear(
            config.n_heads * config.d_head,
            config.d_model,
            bias=False,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.config.n_heads, self.config.d_head)
        xk = xk.view(bsz, seqlen, self.config.n_kv_heads, self.config.d_head)
        xv = xv.view(bsz, seqlen, self.config.n_kv_heads, self.config.d_head)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # ASY: Remove caches
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.config.d_head)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super().__init__()

        self.w1 = nn.Linear(
            config.d_model,
            config.d_ffn,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            config.d_ffn,
            config.d_model,
            bias=False,
            device=device,
        )
        self.w3 = nn.Linear(
            config.d_model,
            config.d_ffn,
            bias=False,
            device=device,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, device):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.d_model
        self.attention = Attention(config, device)
        self.feed_forward = FeedForward(config, device)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.d_model, device=device, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, device=device, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, device):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model, device=device)

        self.layers = torch.nn.ModuleList()

        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config, device=device))

        self.norm = RMSNorm(config.d_model, device=device, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False, device=device)

        self.freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads,
            max_seq_len * 2,
            device=device,
            theta=config.rope_theta,
            use_scaled=False,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)

        output = self.output(h).float()

        return output


def sample_top_p(probs, p):
    """Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
