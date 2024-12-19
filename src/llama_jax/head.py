"""Head."""

from typing import NamedTuple

from jax import Array

from .rms_norm import RMSNorm

__all__ = [
    "Head",
]


class Head(NamedTuple):
    """Head state."""

    norm: RMSNorm

    output: Array
