"""Normalization."""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from llama_jax.checkpoint import ModelConfig, ModelParameters

__all__ = [
    "RMSNorm",
    "create",
    "forward",
]


class RMSNorm(NamedTuple):
    """RMS Normalization state."""

    weight: Array


def create(config: ModelConfig, params: ModelParameters, path: str) -> RMSNorm:
    """Load Llama3 RMSNorm."""
    return RMSNorm(weight=params[f"{path}.weight"])


def forward(config: ModelConfig, state: RMSNorm, x: ArrayLike) -> Array:
    """Normalize x using RMS Normalization.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return state.weight * x * _norm(config, state, x)


def _norm(config: ModelConfig, state: RMSNorm, x: ArrayLike) -> Array:
    """Calculate normalizing factor.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return 1 / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + config.rms_norm_eps)
