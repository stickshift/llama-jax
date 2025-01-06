"""Llama Model."""

from collections.abc import Sequence
from functools import partial
import logging
from typing import NamedTuple

import jax
from jax import Array, random
from jax import numpy as jnp
from jax.nn import softmax

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.kv_cache import KVCache, MutableKVCache
from llama_jax.layer import Layer
from llama_jax.rope import Rope
from llama_jax.tools import default_arg, trace

__all__ = [
    "Model",
    "create",
    "forward",
    "next_token",
]

# Module logger
logger = logging.getLogger(__name__)


class Model(NamedTuple):
    """Model state."""

    embeddings: Embeddings

    layers: Sequence[Layer]

    head: Head

    rope: Rope

    mask: Array


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

    # Rope
    rope = ll.rope.create(config)

    # Mask
    mask = ll.attention.attention_mask(config)

    model = Model(
        embeddings=embeddings,
        layers=layers,
        head=head,
        rope=rope,
        mask=mask,
    )

    return model


@partial(jax.jit, static_argnames=("config",))
def forward(
    config: ModelConfig,
    state: Model,
    token_ids: Array,
    *,
    kv_cache: KVCache | None = None,
) -> Array | tuple[Array, KVCache]:
    """Transform token_ids into next token logits."""
    # Remember if cache was provided
    external_cache = kv_cache is not None

    # Defaults
    kv_cache = default_arg(kv_cache, default_factory=partial(ll.kv_cache.create, config))

    # Sanity check
    assert token_ids.ndim == 2

    # Map tokens to embeddings
    x = ll.embeddings.forward(config, state.embeddings, token_ids)

    # Create mutable kv cache
    kvc = MutableKVCache(kv_cache)

    # Apply layers
    for i, layer in enumerate(state.layers):
        x, kvc[i] = ll.layer.forward(config, layer, state.rope, state.mask, x, kvc[i])

    # Convert kv caches back into immutable sequence
    kv_cache = KVCache(kvc)

    # Apply head
    x = ll.head.forward(config, state.head, x)

    # Return updated cache if provided
    if external_cache:
        return x, kv_cache

    return x


@partial(jax.jit, static_argnames=("temperature", "top_k", "top_p"))
def next_token(
    logits: Array,
    key: Array,
    *,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[Array, Array]:
    """Select next token using temperature, top_p, and top_k sampling.

    Sample tokens from a probability distribution, allowing control over randomness and diversity using the
    temperature, top-k, and top-p parameters.

    Args:
        logits (Array): Next token logits.
        key (Array): RNG key.
        temperature (float): (Optional) The sampling temperature. Higher values (e.g., 1.0) increase randomness, while
            lower values (e.g., 0.1) make output more deterministic. If set to 0, random sampling is disabled and the
            top scoring token is always returned. Defaults to 0.6.
        top_k (int): (Optional) The number of top tokens to consider during sampling. Defaults to 50.
        top_p (float): (Optional) The cumulative probability threshold for nucleus sampling. Only tokens within the top
            cumulative probability of `top_p` are considered. Defaults to 0.9.

    Returns:
        tuple[Array, Array]: Tuple with next token and fresh rng key (next_token, key).
    """
    # Validate
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape {logits.shape}. Expected (bs, vocab_size).")

    # Axes
    batch_axis, value_axis = 0, 1

    # Defaults
    temperature = default_arg(temperature, 0.6)

    # Validate
    if key is None and temperature != 0:
        raise ValueError("Key must be specified unless temperature is 0.")

    # Temperature
    # -----------

    # If temperature is 0, return the top token
    if temperature == 0:
        return jnp.argmax(logits, axis=value_axis, keepdims=True), key

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

    probs = _sample_top_k(probs, top_k=top_k)

    # Top P
    # -----

    probs = _sample_top_p(probs, top_p=top_p)

    # Random Selection
    # ----------------

    # Create separate keys for each batch
    keys = random.split(key, logits.shape[batch_axis] + 1)
    key, subkeys = keys[0], keys[1:]

    # Sample from remaining tokens weighted by probability
    selected = _select_index(probs, key=subkeys)

    # Convert selected index to original logits
    selected = jnp.reshape(selected, (*selected.shape, 1))
    next_token_id = jnp.take_along_axis(indices, selected, axis=value_axis)

    return next_token_id, key


def _sample_top_k(probs: Array, top_k: int | None = None) -> Array:
    # Defaults
    top_k = default_arg(top_k, 50)

    # Sanity check: probs is batched
    assert probs.ndim == 2

    # If there are less than top_k tokens, we're done
    if probs.shape[-1] <= top_k:
        return probs

    # Retain top k tokens
    probs = probs[..., :top_k]

    return probs


def _sample_top_p(probs: Array, top_p: float | None = None) -> Array:
    # Defaults
    top_p = default_arg(top_p, 0.9)

    # Sanity check: probs is batched
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
def _select_index(probs: Array, key: Array) -> Array:
    """Randomly choose index weighted by probability."""
    # Redistribute probs
    probs = probs / jnp.sum(probs)

    # Randomly select index according to p
    pool = jnp.arange(probs.shape[0])
    index = random.choice(key, pool, p=probs)

    return index
