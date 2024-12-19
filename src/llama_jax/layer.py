"""Decoder layer."""

from typing import NamedTuple

from .attention import Attention
from .ffn import FFN

__all__ = [
    "Layer",
]


class Layer(NamedTuple):
    """Decoder layer state."""

    attention: Attention

    ffn: FFN
