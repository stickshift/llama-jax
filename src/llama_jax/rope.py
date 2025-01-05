"""Rotary Position Embeddings (RoPE)."""

from typing import NamedTuple, cast

from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike

from llama_jax.checkpoint import ModelConfig
from llama_jax.tools import recursive_tuple

__all__ = [
    "Rope",
    "RotationMatrix",
    "create",
    "rotate",
    "rotation_matrices",
    "swap",
]

RotationMatrix = tuple[tuple[float, ...]]


def rotation_matrices(base_theta: float, d: int, dtype: DTypeLike, n: int) -> tuple[RotationMatrix, RotationMatrix]:
    """Compute RoPE cos and sin rotation matrices."""
    # Calculate thetas
    i = jnp.arange(d // 2)
    thetas = base_theta ** (-2 * i / d)

    # Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    thetas = jnp.repeat(thetas, 2)

    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = jnp.stack([m * thetas for m in range(n)])

    # Apply cos, sin
    cos = jnp.cos(theta_stack)
    sin = jnp.sin(theta_stack)

    # Convert to dtype after all the upfront math is complete.
    cos, sin = cos.astype(dtype), sin.astype(dtype)

    # Sanity check
    assert cos.shape[0] == n and cos.shape[1] == d  # noqa: PT018
    assert sin.shape[0] == n and sin.shape[1] == d  # noqa: PT018

    # Convert to hashable RotationMatrices
    cos_values = cast(RotationMatrix, recursive_tuple(cos.tolist()))
    sin_values = cast(RotationMatrix, recursive_tuple(sin.tolist()))

    return cos_values, sin_values


class Rope(NamedTuple):
    """RoPE state."""

    cos: Array

    sin: Array


def create(config: ModelConfig, n: int) -> Rope:
    """Compute RoPE cos and sin rotation matrices."""
    return Rope(
        cos=jnp.array(config.rope_cos[:n]),
        sin=jnp.array(config.rope_sin[:n]),
    )


def swap(x: Array) -> Array:
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


def rotate(rope: Rope, x: Array, positions: Array) -> Array:
    """Rotate embeddings using RoPE transform.

    Each pair of values in x is rotated by `m*theta_i`, where m is the position of the embedding in the sequence and `i`
    is the position of the pair in the embedding vector.
    """
    # Select rope entries based on positions
    r_cos, r_sin = rope.cos[positions], rope.sin[positions]

    # Rotate
    x = (x * r_cos) + (swap(x) * r_sin)

    return x
