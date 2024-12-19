"""Llama Model."""

from collections.abc import Sequence
from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike, DTypeLike

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

    rope_theta: float

    d_head: int

    dtype: DTypeLike

    embeddings: Embeddings

    layers: Sequence[Layer]

    head: Head


def create(config: ModelConfig, params: ModelParameters) -> Model:
    """Load Llama3 Model."""
    embeddings = ll.embeddings.create(config, params)

    layers = tuple(
        ll.layer.create(
            config,
            params,
            f"layers.{i}",
        )
        for i in range(config.n_layers)
    )

    head = ll.head.create(config, params)

    return Model(
        rope_theta=config.rope_theta,
        d_head=config.d_head,
        dtype=config.dtype,
        embeddings=embeddings,
        layers=layers,
        head=head,
    )


def forward(state: Model, token_ids: ArrayLike) -> Array:
    """Transform embeddings into token logits."""
    # Sequence length
    n = len(token_ids)

    # RoPE rotation matrices
    r_cos, r_sin = ll.attention.rope_frequencies(
        n,
        base=state.rope_theta,
        d=state.d_head,
        dtype=state.dtype,
    )

    # Masked attention bias
    m = ll.attention.masked_attention_bias(n, state.dtype)

    # Map token ids to embeddings
    x = ll.embeddings.forward(state.embeddings, token_ids)

    # Apply layers
    for layer in state.layers:
        x = ll.layer.forward(layer, x, r_cos, r_sin, m)

    # Apply head
    x = ll.head.forward(state.head, x)

    return x
