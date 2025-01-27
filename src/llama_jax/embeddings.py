"""Embeddings."""

from typing import NamedTuple

from jax import Array

from llama_jax.checkpoint import ModelConfig, ModelParameters

__all__ = [
    "Embeddings",
    "create",
    "forward",
]


class Embeddings(NamedTuple):
    """Embeddings state."""

    values: Array


def create(config: ModelConfig, params: ModelParameters) -> Embeddings:
    """Load Llama3 Embeddings."""
    return Embeddings(values=params["tok_embeddings.weight"].astype(config.dtype))


def forward(config: ModelConfig, state: Embeddings, token_ids: Array) -> Array:
    """Map token ids to embeddings."""
    # Sanity check
    assert token_ids.ndim == 2

    return state.values[token_ids]
