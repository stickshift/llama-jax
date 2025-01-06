from typing import NamedTuple

from jax import Array
from jax import numpy as jnp

from llama_jax.checkpoint import ModelConfig

__all__ = [
    "KVCache",
    "LayerKVCache",
    "MutableKVCache",
    "apply",
    "create",
]


class LayerKVCache(NamedTuple):
    """Key-Value cache for single attention layer.

    Each array has shape (bs, n_heads, n, d_head).
    """

    keys: Array | None = None

    values: Array | None = None


KVCache = tuple[LayerKVCache, ...]
MutableKVCache = list[LayerKVCache]


def create(config: ModelConfig) -> KVCache:
    """Create key-value cache for attention layers."""
    return tuple(LayerKVCache() for _ in range(config.n_layers))


def apply(cached_values: Array | None, values: Array) -> Array:
    """Add values to cache and return entire cache."""
    if cached_values is None:
        return values

    return jnp.concat([cached_values, values], axis=-2)
