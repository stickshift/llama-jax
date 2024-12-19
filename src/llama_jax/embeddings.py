"""Embeddings."""

from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "Embeddings",
]


class Embeddings(NamedTuple):
    """Embeddings state."""

    values: Array


def embeddings(state: Embeddings, token_ids: ArrayLike) -> Array:
    """Map token ids to embeddings."""
    return state.values[token_ids]
