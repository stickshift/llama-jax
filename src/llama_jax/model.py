"""Llama Model."""

from collections.abc import Sequence
from functools import partial
from typing import NamedTuple

import jax
from jax import Array, random
from jax import numpy as jnp
from jax.nn import softmax
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.kv_cache import KVCache, MutableKVCache
from llama_jax.layer import Layer
from llama_jax.tools import default_arg

__all__ = [
    "Model",
    "create",
    "forward",
]


class Model(NamedTuple):
    """Model state."""

    embeddings: Embeddings

    layers: Sequence[Layer]

    head: Head


def create(config: ModelConfig, params: ModelParameters) -> Model:
    """Load Llama3 Model."""
    # Embeddings
    embeddings = ll.embeddings.create(config, params)

    # Layers
    layers = tuple(
        ll.layer.create(
            config,
            params,
            f"layers.{i}",
        )
        for i in range(config.n_layers)
    )

    # Head
    head = ll.head.create(config, params)

    return Model(
        embeddings=embeddings,
        layers=layers,
        head=head,
    )


# @partial(jax.jit, static_argnames=("config",))
def forward(
    config: ModelConfig,
    state: Model,
    kv_cache: KVCache,
    token_ids: ArrayLike,
) -> tuple[Array, KVCache]:
    """Transform token ids into next token logits."""
    # Sanity check
    assert token_ids.ndim == 2

    # Sequence length
    n = token_ids.shape[-1]

    # RoPE rotation matrices
    rope = ll.rope.create(config, n)

    # Masked attention bias
    mask = ll.attention.attention_mask(n, config.dtype)

    # Map token ids to embeddings
    x = ll.embeddings.forward(config, state.embeddings, token_ids)

    # Make kv caches mutable
    kv_cache = MutableKVCache(kv_cache)

    # Apply layers
    for i, layer in enumerate(state.layers):
        x, kv_cache[i] = ll.layer.forward(config, layer, rope, mask, kv_cache[i], x)

    # Convert kv caches back into immutable sequence
    kv_cache = KVCache(kv_cache)

    # Apply head
    x = ll.head.forward(config, state.head, x)

    return x, kv_cache


@partial(jax.jit, static_argnames=("temperature", "top_k", "top_p"))
def sample_tokens(
    logits: ArrayLike,
    *,
    key: ArrayLike,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Array:
    """Select next token using temperature, top_p, and top_k sampling."""
    # Sanity check
    assert logits.ndim == 2

    # Axes - logits shape: (bs, vocab_size)
    batch_axis, value_axis = 0, 1
    n_batches = logits.shape[batch_axis]

    # Defaults
    temperature = default_arg(temperature, 0.6)

    # Temperature
    # -----------

    # If temperature is 0, return the top token
    if temperature == 0:
        return jnp.argmax(logits, axis=value_axis, keepdims=True)

    # Apply temperature
    logits = logits / temperature

    # Ranking
    # -------

    # Convert logits to probabilities
    probs = softmax(logits, axis=value_axis)

    # Sort probabilities in descending order, maintaining original indices
    indices = jnp.argsort(probs, axis=value_axis, descending=True)
    probs = jnp.take_along_axis(probs, indices, axis=value_axis)

    # Top K
    # -----

    probs = sample_top_k(probs, top_k=top_k)

    # Top P
    # -----

    probs = sample_top_p(probs, top_p=top_p)

    # Random Selection
    # ----------------

    # Sample from remaining tokens weighted by probability
    keys = random.split(key, n_batches + 1)
    key, subkeys = keys[0], keys[1:]
    selected = select_index(probs, key=subkeys)

    # Convert selected index to original logits
    selected = jnp.reshape(selected, (*selected.shape, 1))
    next_token_id = jnp.take_along_axis(indices, selected, axis=value_axis)

    return next_token_id


def sample_top_k(probs: ArrayLike, top_k: int | None = None) -> Array:
    # Defaults
    top_k = default_arg(top_k, 50)

    # Retain top k tokens
    probs = probs[..., :top_k]

    # Sanity check
    assert probs.shape[-1] == top_k

    return probs


def sample_top_p(probs: ArrayLike, top_p: float | None = None) -> Array:
    # Defaults
    top_p = default_arg(top_p, 0.9)

    # Sanity check
    assert probs.ndim == 2

    # Axes: (bs, vocab_size)
    batch_axis, value_axis = 0, 1
    n_batches, n_values = probs.shape[batch_axis], probs.shape[value_axis]

    # Find cutoff where cumulative probability exceeds top_p
    cumulative_mask = probs.cumsum(axis=value_axis) >= top_p
    threshold = jnp.argmax(cumulative_mask, axis=value_axis, keepdims=True)

    # Zero out probs above threshold
    indices = jnp.reshape(jnp.arange(n_values), (1, n_values))
    indices = jnp.repeat(indices, n_batches, axis=0)
    mask = (indices <= threshold).astype(jnp.int32)
    probs = probs * mask

    return probs


@jax.vmap
def select_index(probs: ArrayLike, key: ArrayLike) -> Array:
    """Randomly choose index weighted by probability."""
    index = random.choice(key, jnp.arange(probs.shape[0]), p=probs)

    return index
