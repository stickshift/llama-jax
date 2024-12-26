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

    weight = params[f"{path}.weight"]

    # Convert weights to float32.
    #   ASY - We're recreating behavior from llama-models I believe to be a bug. Even though the rms weights are saved
    #   in bfloat16, they're always initialized as float32.
    #
    weight = weight.astype(jnp.float32)

    return RMSNorm(weight=weight)


def forward(config: ModelConfig, state: RMSNorm, x: ArrayLike) -> Array:
    """Normalize x using RMS Normalization.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    # Sanity check
    assert x.ndim == 3

    return state.weight * _norm(config, state, x.astype(jnp.float32)).astype(x.dtype)


def _norm(config: ModelConfig, state: RMSNorm, x: ArrayLike) -> Array:
    """Calculate normalizing factor.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + config.rms_norm_eps)
