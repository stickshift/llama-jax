"""Attention."""

from typing import NamedTuple

from jax import Array

from .normalization import RMSNorm

__all__ = [
    "Attention",
]


class Attention(NamedTuple):
    """Attention state."""

    n_heads: int

    n_kv_heads: int

    d_head: int

    norm: RMSNorm

    queries: Array

    keys: Array

    values: Array

    output: Array
