"""Normalization."""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "RMSNorm",
    "rms",
]


class RMSNorm(NamedTuple):
    """RMS Normalization state."""

    weight: Array

    eps: float


def rms(state: RMSNorm, x: ArrayLike) -> Array:
    """Normalize x using RMS Normalization.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return state.weight * x / jnp.sqrt(jnp.mean(x**2) + state.eps)
