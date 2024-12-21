"""Rotary Position Embeddings (RoPE)."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from llama_jax.checkpoint import ModelConfig

__all__ = [
    "Rope",
    "create",
    "swap",
    "rotate",
]


class Rope(NamedTuple):
    """RoPE state."""

    cos: Array

    sin: Array


def create(config: ModelConfig, n: int) -> Rope:
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
    assert r_cos.shape[0] == n and r_cos.shape[1] == d  # noqa: PT018
    assert r_sin.shape[0] == n and r_sin.shape[1] == d  # noqa: PT018

    return Rope(cos=r_cos, sin=r_sin)


def swap(x: ArrayLike) -> Array:
    """Maps [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] along last dimension."""
    # Split last dimension into pairs
    y = jnp.reshape(x, (-1, 2))

    # Swap pairs. e.g. [x0, x1] -> [x1, x0]
    y = y[:, ::-1]

    # Restore original shape
    y = jnp.reshape(y, x.shape)

    # Create a mask for even indices along the last dimension
    mask = jnp.arange(y.shape[-1]) % 2 == 0

    # Apply the mask to multiply even indices by -1
    #   e.g. [x0, x1, x2, x3] -> [-x0, x1, -x2, x3]
    y = y * jnp.where(mask, -1, 1)

    return y


def rotate(rope: Rope, x: ArrayLike) -> Array:
    """Rotate embeddings using RoPE transform."""
    return (x * rope.cos) + (swap(x) * rope.sin)
