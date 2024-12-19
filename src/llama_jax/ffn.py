"""Feedforward Network."""

from typing import NamedTuple

from jax import Array

from .normalization import RMSNorm

__all__ = [
    "FFN",
]


class FFN(NamedTuple):
    """Feedforward Network state."""

    norm: RMSNorm

    input: Array

    gate: Array

    output: Array
