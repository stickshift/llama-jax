"""Embeddings."""

from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike

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
    return Embeddings(values=params["tok_embeddings.weight"])


def forward(state: Embeddings, token_ids: ArrayLike) -> Array:
    """Map token ids to embeddings."""
    return state.values[token_ids]
