"""Normalization."""

from typing import NamedTuple

from jax import Array
import jax.numpy as jnp

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
    return RMSNorm(weight=params[f"{path}.weight"].astype(config.dtype))


def forward(config: ModelConfig, state: RMSNorm, x: Array) -> Array:
    """Normalize x using RMS Normalization.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    # Validate
    if x.ndim != 3:
        raise ValueError(f"Unexpected shape {x.shape}. Expected (bs, n, d).")

    return state.weight * _norm(x, config.rms_norm_eps)


def _norm(x: Array, rms_norm_eps: float) -> Array:
    """Calculate normalizing factor.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + rms_norm_eps)
