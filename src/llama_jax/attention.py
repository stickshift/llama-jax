"""Attention."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

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

    return Attention(
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        d_head=config.d_head,
        norm=ll.rms_norm.create(config, params, f"{parent_path}.attention_norm"),
        queries=params[f"{path}.wq.weight"],
        keys=params[f"{path}.wk.weight"],
        values=params[f"{path}.wv.weight"],
        output=params[f"{path}.wo.weight"],
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

#
# def masked_attention_bias(config: ModelConfig, n: int) -> Array:
#     """Compute reusable masked attention bias term M.
#
#     Returns:
#         Tensor: (n, n) diagonal matrix w/ upper triangular elements set to -inf, 0 otherwise.
#     """
#     from torch import logical_not, ones, tril, zeros  # noqa: PLC0415
#
#     # Parameters
#     dtype = config.dtype
#
#     # Initialize m with zeros
#     m = zeros(n, n, device=device, dtype=dtype)
#
#     # Create boolean mask w/ main diagonal and below set to False
#     mask = logical_not(tril(ones(n, n, device=device, dtype=torch.bool)))
#
#     # Fill upper triangular region to -inf
#     m = m.masked_fill_(mask, float("-inf"))
#
#     return m


def forward(state: Attention, x: ArrayLike) -> Array:
    """Transform x using grouped query attention (GQA)."""
    # Save residuals
    residual = x

    # Normalize inputs
    x = ll.rms_norm.forward(state.norm, x)

    # Project inputs to query, key, value spaces
    q = x @ jnp.transpose(state.queries)
    k = x @ jnp.transpose(state.keys)
    v = x @ jnp.transpose(state.values)

    # Split attention heads
    q = _split_heads(q, state.n_heads)
    k = _split_heads(k, state.n_kv_heads)
    v = _split_heads(v, state.n_kv_heads)

    # Expand key/value groups
    reps = state.n_heads // state.n_kv_heads
    k = k.repeat_interleave(reps, dim=0)
    v = v.repeat_interleave(reps, dim=0)

    # Encode positions by rotating queries and keys
    q = rope_rotate(q, r_cos, r_sin)
    k = rope_rotate(k, r_cos, r_sin)

    # Compute attention for all heads in parallel
    scores = q @ k.transpose(-2, -1) / np.sqrt(self.config.d_head) + mask
    a = F.softmax(scores, dim=-1) @ v

    # Combine attention heads
    a = self._combine_heads(a)

    # Project outputs back to model space
    a = self.w_output(a)

    # Merge outputs with residuals
    x = residual + a

    return x
