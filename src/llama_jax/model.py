"""Llama Model."""

from collections.abc import Sequence
from functools import partial
from typing import NamedTuple

import jax
from jax import Array
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer

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


class ModelOutput(NamedTuple):
    """Model output."""

    logits: Array


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


@partial(jax.jit, static_argnames=("config",))
def forward(
    config: ModelConfig,
    state: Model,
    token_ids: ArrayLike,
) -> ModelOutput:
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

    # Apply layers
    for i, layer in enumerate(state.layers):
        x = ll.layer.forward(config, layer, rope, mask, x)

    # Apply head
    x = ll.head.forward(config, state.head, x)

    return ModelOutput(logits=x)
