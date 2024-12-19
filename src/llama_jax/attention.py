"""Attention."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.nn import softmax
from jax.typing import ArrayLike, DTypeLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "Attention",
    "create",
    "forward",
    "rope_frequencies",
    "rope_rotate",
    "rope_swap",
]


class Attention(NamedTuple):
    """Attention state."""

    n_heads: int

    n_kv_heads: int

    d_head: int

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
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        d_head=config.d_head,
        norm=ll.rms_norm.create(config, params, f"{parent_path}.attention_norm"),
        queries=queries,
        keys=keys,
        values=values,
        output=output,
    )


def rope_frequencies(config: ModelConfig, n: int) -> tuple[Array, Array]:
    """Compute RoPE cos and sin rotation matrices."""
    # Hyperparameters
    base = config.rope_theta
    d = config.d_head
    dtype = config.dtype

    # Calculate thetas
    i = jnp.arange(d // 2, dtype=dtype)
    thetas = base ** (-2 * i / d)

    # Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    thetas = jnp.repeat(thetas, 2)

    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = jnp.stack([m * thetas for m in range(n)])

    # Apply cos, sin
    r_cos = jnp.cos(theta_stack)
    r_sin = jnp.sin(theta_stack)

    # Sanity check
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head  # noqa: PT018
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head  # noqa: PT018

    return r_cos, r_sin


def rope_swap(x: ArrayLike) -> Array:
    """Maps [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] along last dimension."""
    # Split last dimension into pairs
    y = x.reshape(-1, 2)

    # Swap pairs. e.g. [x0, x1] -> [x1, x0]
    y = y[:, ::-1]

    # Restore original shape
    y = y.reshape(x.shape)

    # Create a mask for even indices along the last dimension
    mask = jnp.arange(y.shape[-1]) % 2 == 0

    # Apply the mask to multiply even indices by -1
    #   e.g. [x0, x1, x2, x3] -> [-x0, x1, -x2, x3]
    y = y * jnp.where(mask, -1, 1)

    return y


def rope_rotate(x, r_cos, r_sin):
    """Rotate embeddings using RoPE transform."""
    return (x * r_cos) + (rope_swap(x) * r_sin)


def masked_attention_bias(n: int, dtype: DTypeLike) -> Array:
    """Compute reusable masked attention bias term M.

    Returns:
        Array: (n, n) diagonal matrix w/ upper triangular elements set to -inf, 0 otherwise.
    """
    # Create boolean mask w/ main diagonal and below set to False
    mask = ~jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))

    # Apply mask to fill array with -inf, 0 otherwise
    m = jnp.where(mask, -jnp.inf, 0)

    return m.astype(dtype)


def split_heads(x: Array, n_heads: int) -> Array:
    """Split attention heads."""
    # Sanity check
    assert len(x.shape) >= 2

    # Split last dimension into n_heads groups:
    #   e.g (..., n, d_model) -> (..., n, n_heads, d_head)
    d_head = x.shape[-1] // n_heads
    y = x.reshape(-1, n_heads, d_head)

    # Transpose dimensions -3 and -2
    #   e.g (..., n, n_heads, d_head) -> (..., n_heads, n, d_head)
    y = y.swapaxes(-3, -2)

    return y


def combine_heads(x: Array) -> Array:
    """Combine attention heads."""
    # Sanity check
    assert len(x.shape) >= 3

    d_head = x.shape[-1]
    n_heads = x.shape[-3]

    # Transpose dimensions -3 and -2
    #   e.g (..., n_heads, n, d_head) -> (..., n, n_heads, d_head)
    y = x.swapaxes(-3, -2)

    # Merge last 2 dimensions
    #   e.g (..., n, n_heads, d_head) -> (..., n, n_heads*d_head)
    y = y.reshape(-1, n_heads * d_head)

    return y


def forward(state: Attention, x: ArrayLike, r_cos: ArrayLike, r_sin: ArrayLike, mask: ArrayLike) -> Array:
    """Transform x using grouped query attention (GQA)."""
    # Save residuals
    residual = x

    # Normalize inputs
    x = ll.rms_norm.forward(state.norm, x)

    # Project inputs to query, key, value spaces
    q = x @ state.queries
    k = x @ state.keys
    v = x @ state.values

    # Split attention heads
    q = split_heads(q, state.n_heads)
    k = split_heads(k, state.n_kv_heads)
    v = split_heads(v, state.n_kv_heads)

    # Expand key/value groups
    reps = state.n_heads // state.n_kv_heads
    k = k.repeat(reps, axis=0)
    v = v.repeat(reps, axis=0)

    # Encode positions by rotating queries and keys
    q = rope_rotate(q, r_cos, r_sin)
    k = rope_rotate(k, r_cos, r_sin)

    # Compute attention for all heads in parallel
    #   e.g. softmax((Q * K^T) / sqrt(d_head) + M) * V
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(state.d_head) + mask
    x = softmax(scores, axis=-1) @ v

    # Combine attention heads
    x = combine_heads(x)

    # Project outputs back to model space
    x = x @ state.output

    # Merge outputs with residuals
    x = residual + x

    return x
