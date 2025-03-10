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
from llama_jax.kvc import KVCache
from llama_jax.layer import Layer
from llama_jax.rope import Rope
from llama_jax.tools import default_arg

__all__ = [
    "Model",
    "create",
    "forward",
    "next_token_id",
]

# Module logger
logger = logging.getLogger(__name__)


class Model(NamedTuple):
    """Model state."""

    embeddings: Embeddings

    layers: Sequence[Layer]

    head: Head

    rope: Rope


def create(config: ModelConfig, params: ModelParameters | None = None) -> Model:
    """Create a Llama 3 model.

    Args:
        config (ModelConfig): Checkpoint configuration.
        params (ModelParameters): (Optional) Override model parameters.

    Returns:
        Model state.
    """
    # Defaults
    params = default_arg(params, default_factory=partial(ll.checkpoint.load_parameters, config))

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

    model = Model(
        embeddings=embeddings,
        layers=layers,
        head=head,
        rope=rope,
    )

    return model


@partial(jax.jit, static_argnames=("config",))
def forward(
    config: ModelConfig,
    state: Model,
    token_ids: Array,
    position_mask: Array,
    *,
    kvc: KVCache | None = None,
) -> Array | tuple[Array, KVCache]:
    """Transform token_ids into next token logits."""
    # Validate
    if token_ids.shape[-1] > config.max_sequence_length:
        raise ValueError(f"Number of tokens exceed config.max_tokens {config.max_sequence_length}")

    # Remember if cache was provided
    external_cache = kvc is not None

    # Defaults
    kvc = default_arg(kvc, default_factory=partial(ll.kvc.create, config))

    # Sanity check
    assert token_ids.ndim == 2

    # Map tokens to embeddings
    x = ll.embeddings.forward(config, state.embeddings, token_ids)

    # Create mask
    mask = ll.attention.attention_mask(config, position_mask)

    # Create mutable kv cache
    kvc_layers = list(kvc)

    # Apply layers
    for i, layer in enumerate(state.layers):
        x, kvc_layers[i] = ll.layer.forward(config, layer, state.rope, mask, x, kvc_layers[i])

    # Convert kv caches back into immutable sequence
    kvc = KVCache(kvc_layers)

    # Apply head
    x = ll.head.forward(config, state.head, x, position_mask)

    # Return updated cache if provided
    if external_cache:
        return x, kvc

    return x


@partial(jax.jit, static_argnames=("temperature", "top_k", "top_p"))
def next_token_id(
    logits: Array,
    *,
    key: Array | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Array:
    """Select next token id using temperature, top k, and top p sampling."""
    # Validate
    if logits.ndim != 2:
        raise ValueError(f"Unexpected logits shape {logits.shape}. Expected (bs, vocab_size).")

    if key is None and temperature != 0:
        raise ValueError("Key must be specified unless temperature is 0.")

    # Temperature
    # -----------

    # Defaults
    temperature = default_arg(temperature, 0.6)

    # If temperature is 0, return the top token
    if temperature == 0:
        return jnp.argmax(logits, axis=-1, keepdims=True)

    # Apply temperature
    logits = logits / temperature

    # Ranking
    # -------

    # Sort logits in descending order, maintaining original indices
    indices = jnp.argsort(logits, axis=-1, descending=True)

    # Top K
    # -----

    # Defaults
    top_k = default_arg(top_k, 50)

    # Apply top k to entire batch at once
    indices = indices[:, :top_k]
    logits = jnp.take_along_axis(logits, indices, axis=-1)

    # Top P
    # -----

    # Defaults
    top_p = default_arg(top_p, 0.9)

    # Convert remaining logits to probabilities
    probs = softmax(logits, axis=-1)

    # Find index where cumulative sum of probs exceeds p
    cumulative_mask = probs.cumsum(axis=-1) <= top_p
    cutoff = jnp.sum(cumulative_mask, axis=-1, keepdims=True)

    # Calculate mask for indicies <= cutoff
    mask = jnp.broadcast_to(jnp.arange(logits.shape[-1]), logits.shape) <= cutoff

    # Zero out logits above cutoff
    logits = jnp.where(mask, logits, 0)

    # Random Selection
    # ----------------

    assert key is not None

    # Randomly choose from remaining logits
    key, subkey = random.split(key)
    selected = random.categorical(subkey, logits, axis=-1)[:, None]

    # Map selected back to original logit indices
    next_token_id = jnp.take_along_axis(indices, selected, axis=-1)

    return next_token_id
