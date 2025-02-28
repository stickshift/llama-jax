from typing import NamedTuple

import jax
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
    """Key-Value cache for single attention layer."""

    keys: Array | None

    values: Array | None


KVCache = tuple[LayerKVCache, ...]
MutableKVCache = list[LayerKVCache]


def create(config: ModelConfig) -> KVCache:
    """Create key-value cache for attention layers."""

    # All layers start with same layer kvc
    layer_kvc = LayerKVCache(keys=None, values=None)

    return tuple(layer_kvc for _ in range(config.n_layers))


def apply(layer_kvc: LayerKVCache, *, keys: Array, values: Array) -> tuple[LayerKVCache, Array, Array]:
    """Add keys and values to layer cache and return all keys and values in the cache.

    Args:
        layer_kvc (LayerKVCache): layer key value cache
        keys (Array): keys array of shape (bs, n_heads, n, d_head)
        values (Array): values array of shape (bs, n_heads, n, d_head)

    Returns:
        (tuple): Updated layer cache, keys, and values.
    """
    assert keys.shape == values.shape

    # Append keys and values to cache along axis 2
    layer_kvc = LayerKVCache(
        keys=_apply(layer_kvc.keys, keys),
        values=_apply(layer_kvc.values, values),
    )

    return layer_kvc, layer_kvc.keys, layer_kvc.values


def _apply(cached_values: Array | None, values: Array) -> Array:
    """Add values to cache and return entire cache."""
    if cached_values is None:
        return values

    return jnp.concat([cached_values, values], axis=-2)
