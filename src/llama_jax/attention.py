"""Attention."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.nn import softmax

import llama_jax as ll
from llama_jax.checkpoint import HEAD_AXIS, MODEL_AXIS, TOKEN_AXIS, ModelConfig, ModelParameters
from llama_jax.kv_cache import LayerKVCache
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "Attention",
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


def attention_mask(config: ModelConfig, n: int) -> Array:
    """Create (n, n) causal attention mask."""
    return jnp.array([row[:n] for row in config.mask[:n]])


def forward(
    config: ModelConfig,
    state: Attention,
    x: Array,
    kv_cache: LayerKVCache,
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
    k = ll.kv_cache.apply(kv_cache.keys, k)
    v = ll.kv_cache.apply(kv_cache.values, v)
    kv_cache = LayerKVCache(keys=k, values=v)

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
    k_positions = jnp.arange(n_keys)
    q_positions = k_positions[-n_queries:]

    # Encode positions by rotating queries and keys
    rope = ll.rope.create(config, n_keys)
    q = ll.rope.rotate(rope, q, positions=q_positions)
    k = ll.rope.rotate(rope, k, positions=k_positions)

    # Generate mask
    m = attention_mask(config, q.shape[TOKEN_AXIS])

    # Compute attention for all heads in parallel
    #   e.g. softmax((Q * K^T) / sqrt(d_head) + M) * V
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(config.d_head) + m
    x = softmax(scores, axis=-1) @ v

    # Combine attention heads
    #   e.g. (bs, n_heads, n, d_head) -> (bs, n, d_model)
    x = combine_heads(x)

    # Project outputs back to model space
    x = x @ state.output

    # Merge outputs with residuals
    x = residual + x

    return x, kv_cache
