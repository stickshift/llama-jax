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

    n: Array

    key_cache: Array

    value_cache: Array


KVCache = tuple[LayerKVCache, ...]
MutableKVCache = list[LayerKVCache]


def create(config: ModelConfig, bs: int) -> KVCache:
    """Create key-value cache for attention layers."""
    # Initialize keys and values with (bs, n_heads, max_tokens, d_head) buffer
    buffer = jnp.zeros((bs, config.n_kv_heads, config.max_tokens, config.d_head), dtype=config.dtype)

    # All layers start with same layer kvc
    layer_kvc = LayerKVCache(n=jnp.array(0), key_cache=buffer, value_cache=buffer)

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

    n = layer_kvc.n + keys.shape[-2]
    key_cache = jax.lax.dynamic_update_slice(layer_kvc.key_cache, keys, (0, 0, layer_kvc.n, 0))
    value_cache = jax.lax.dynamic_update_slice(layer_kvc.value_cache, values, (0, 0, layer_kvc.n, 0))

    layer_kvc = LayerKVCache(n=n, key_cache=key_cache, value_cache=value_cache)

    keys = layer_kvc.key_cache[:, :, : layer_kvc.n, :]
    values = layer_kvc.value_cache[:, :, : layer_kvc.n, :]

    return layer_kvc, keys, values


# def apply(cached_values: Array | None, values: Array) -> Array:
#     """Add values to cache and return entire cache."""
#     if cached_values is None:
#         return values

#     return jnp.concat([cached_values, values], axis=-2)
