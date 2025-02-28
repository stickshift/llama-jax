"""Attention."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.nn import softmax

import llama_jax as ll
from llama_jax.checkpoint import HEAD_AXIS, MODEL_AXIS, TOKEN_AXIS, ModelConfig, ModelParameters
from llama_jax.kvc import LayerKVCache
from llama_jax.rms_norm import RMSNorm
from llama_jax.rope import Rope

__all__ = [
    "Attention",
    "attention",
    "attention_mask",
    "combine_heads",
    "create",
    "forward",
    "split_heads",
]


# ------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------


class Attention(NamedTuple):
    """Attention state."""

    norm: RMSNorm

    queries: Array

    keys: Array

    values: Array

    output: Array


def create(config: ModelConfig, params: ModelParameters, path: str) -> Attention:
    """Load Llama3 Attention."""
    parent_path = path.rsplit(".", 1)[0]

    # Note we transpose kernels so we don't need to during forward pass
    queries = params[f"{path}.wq.weight"].transpose().astype(config.dtype)
    keys = params[f"{path}.wk.weight"].transpose().astype(config.dtype)
    values = params[f"{path}.wv.weight"].transpose().astype(config.dtype)
    output = params[f"{path}.wo.weight"].transpose().astype(config.dtype)

    return Attention(
        norm=ll.rms_norm.create(config, params, f"{parent_path}.attention_norm"),
        queries=queries,
        keys=keys,
        values=values,
        output=output,
    )


def split_heads(x: Array, n_heads: int) -> Array:
    """Split attention heads.

    Args:
        x (Array): Input tensor w/ shape (bs, n, d_model)
        n_heads (int): Number of attention heads

    Returns:
        Array: x reshaped to (bs, n_heads, n, d_head)
    """
    # Sanity check
    assert len(x.shape) == 3

    # Calculate dimensions from static shape
    d_head = x.shape[MODEL_AXIS] // n_heads

    # Split last dimension into n_heads groups:
    #   e.g (bs, n, d_model) -> (bs, n, n_heads, d_head)
    shape = x.shape[:-1] + (n_heads, d_head)
    y = jnp.reshape(x, shape)

    # Swap token and head dimensions
    #   e.g (bs, n, n_heads, d_head) -> (bs, n_heads, n, d_head)
    y = jnp.swapaxes(y, HEAD_AXIS, TOKEN_AXIS)

    return y


def combine_heads(x: Array) -> Array:
    """Combine attention heads.

    Args:
        x (Array): Input tensor w/ shape (bs, n_heads, n, d_head)

    Returns:
        Array: x reshaped to (bs, n, n_heads * d_head)
    """
    # Sanity check
    assert len(x.shape) == 4

    # Calculate dimensions from static shape
    n_heads, d_head = x.shape[HEAD_AXIS], x.shape[MODEL_AXIS]

    # Swap token and head dimensions
    #   e.g (bs, n_heads, n, d_head) -> (bs, n, n_heads, d_head)
    y = jnp.swapaxes(x, HEAD_AXIS, TOKEN_AXIS)

    # Merge last 2 dimensions
    #   e.g (bs, n, n_heads, d_head) -> (bs, n, n_heads * d_head)
    shape = y.shape[:-2] + (n_heads * d_head,)
    y = jnp.reshape(y, shape)

    return y


def attention_mask(config: ModelConfig, position_mask: Array) -> Array:
    """Compute attention mask."""
    # Sanity check
    assert position_mask.dtype == jnp.int32
    assert position_mask.shape[-1] == config.max_tokens

    # Start with (max_tokens, max_tokens) causal mask
    causal_mask = jnp.tril(jnp.ones((config.max_tokens, config.max_tokens), dtype=jnp.int32))

    # Combine masks: m[b, i, j] = causal_mask[i, j] AND position_mask[b, j]
    #   1) Broadcast causal_mask from (n, n) to (bs, n, n)
    #   2) Broadcast position mask from (bs, n) to (bs, n, n)
    #   3) Logically AND them together

    m = causal_mask[None, :, :] & position_mask[:, None, :]

    # Convert booleans to 0s and -infs
    m = jnp.where(m, 0, -jnp.inf)

    return m


def attention(config: ModelConfig, q: Array, k: Array, v: Array, m: Array) -> Array:
    """Compute attention in parallel across all heads."""
    # Sanity check
    assert q.ndim == k.ndim == v.ndim == m.ndim

    # Attention scores
    scores = softmax(q @ k.swapaxes(-2, -1) / jnp.sqrt(config.d_head) + m, axis=-1)

    # Apply scores to values
    a = scores @ v

    return a


def forward(
    config: ModelConfig,
    state: Attention,
    rope: Rope,
    mask: Array,
    x: Array,
    layer_kvc: LayerKVCache,
) -> tuple[Array, LayerKVCache]:
    """Transform x using grouped query attention (GQA)."""
    # Save residuals
    residual = x

    # Normalize inputs
    x = ll.rms_norm.forward(config, state.norm, x)

    # Project inputs to query, key, value spaces
    q = x @ state.queries
    k = x @ state.keys
    v = x @ state.values

    # Split attention heads
    #   e.g. (bs, n, d_model) -> (bs, n_heads, n, d_head)
    q = split_heads(q, config.n_heads)
    k = split_heads(k, config.n_kv_heads)
    v = split_heads(v, config.n_kv_heads)

    # Update key/value cache
    layer_kvc, k, v = ll.kvc.apply(layer_kvc, keys=k, values=v)

    # Expand key/value groups
    reps = config.n_heads // config.n_kv_heads
    k = k.repeat(reps, axis=HEAD_AXIS)
    v = v.repeat(reps, axis=HEAD_AXIS)

    # Calculate embedding positions in the sequence
    #   * Keys are positioned at [0, ..., n_keys-1]
    #
    #   * Queries are positioned at last n_queries elements. This supports both full sequence mode where
    #     n_queries == n_keys and incremental mode where previous keys are loaded from kv_cache.
    #
    n_queries, n_keys = q.shape[TOKEN_AXIS], k.shape[TOKEN_AXIS]
    q_positions = jnp.arange(n_keys - n_queries, n_keys)
    k_positions = jnp.arange(n_keys)

    # Encode positions by rotating queries and keys
    q = ll.rope.rotate(rope, q, positions=q_positions)
    k = ll.rope.rotate(rope, k, positions=k_positions)

    # Generate (bs, q, k) mask bias term
    m = mask[:, (n_keys - n_queries) : n_keys, :n_keys]

    # Broadcast m to (bs, 1, q, k) to be compatible with split q, k, v
    m = m[:, None, :, :]

    # Compute attention for all heads in parallel
    x = attention(config, q, k, v, m)

    # Combine attention heads
    #   e.g. (bs, n_heads, n, d_head) -> (bs, n, d_model)
    x = combine_heads(x)

    # Project outputs back to model space
    x = x @ state.output

    # Merge outputs with residuals
    x = residual + x

    return x, layer_kvc
