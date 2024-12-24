"""Attention."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.nn import softmax
from jax.typing import ArrayLike, DTypeLike

import llama_jax as ll
from llama_jax.checkpoint import HEAD_AXIS, MODEL_AXIS, TOKEN_AXIS, ModelConfig, ModelParameters
from llama_jax.kv_cache import LayerKVCache
from llama_jax.rms_norm import RMSNorm
from llama_jax.rope import Rope

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
    queries = params[f"{path}.wq.weight"].transpose()
    keys = params[f"{path}.wk.weight"].transpose()
    values = params[f"{path}.wv.weight"].transpose()
    output = params[f"{path}.wo.weight"].transpose()

    return Attention(
        norm=ll.rms_norm.create(config, params, f"{parent_path}.attention_norm"),
        queries=queries,
        keys=keys,
        values=values,
        output=output,
    )


def attention_mask(n: int, dtype: DTypeLike) -> Array:
    """Compute reusable masked attention bias term M.

    Returns:
        Array: (n, n) diagonal matrix w/ upper triangular elements set to -inf, 0 otherwise.
    """
    # Create boolean mask w/ main diagonal and below set to False
    mask = ~jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))

    # Apply mask to fill array with -inf, 0 otherwise
    m = jnp.where(mask, -jnp.inf, 0)

    return m.astype(dtype)


def split_heads(x: ArrayLike, n_heads: int) -> Array:
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


def forward(
    config: ModelConfig,
    state: Attention,
    rope: Rope,
    mask: ArrayLike,
    kv_cache: LayerKVCache,
    x: ArrayLike,
) -> tuple[Array, LayerKVCache]:
    """Transform x using grouped query attention (GQA)."""
    # Sanity check
    assert x.ndim == 3

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

    # Expand key/value groups along n_heads dimension
    reps = config.n_heads // config.n_kv_heads
    k = k.repeat(reps, axis=HEAD_AXIS)
    v = v.repeat(reps, axis=HEAD_AXIS)

    # Encode positions by rotating queries and keys
    q = ll.rope.rotate(rope, q)
    k = ll.rope.rotate(rope, k)

    # Compute attention for all heads in parallel
    #   e.g. softmax((Q * K^T) / sqrt(d_head) + M) * V
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(config.d_head) + mask
    x = softmax(scores, axis=-1) @ v

    # Combine attention heads
    #   e.g. (bs, n_heads, n, d_head) -> (bs, n, d_model)
    x = combine_heads(x)

    # Project outputs back to model space
    x = x @ state.output

    # Merge outputs with residuals
    x = residual + x

    return x, kv_cache
