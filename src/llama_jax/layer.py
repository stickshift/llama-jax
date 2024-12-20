"""Decoder layer."""

from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.ffn import FFN

__all__ = [
    "Layer",
    "create",
    "forward",
]


class Layer(NamedTuple):
    """Decoder layer state."""

    attention: Attention

    ffn: FFN


def create(config: ModelConfig, params: ModelParameters, path: str) -> Layer:
    """Load Llama3 Layer."""
    # Attention
    attention = ll.attention.create(config, params, f"{path}.attention")

    # FFN
    ffn = ll.ffn.create(config, params, f"{path}.feed_forward")

    return Layer(attention=attention, ffn=ffn)


def forward(state: Layer, x: ArrayLike, r_cos: ArrayLike, r_sin: ArrayLike, mask: ArrayLike) -> Array:
    """Transform x using attention and feedforward network."""
    # Attention
    x = ll.attention.forward(state.attention, x, r_cos, r_sin, mask)

    # FFN
    x = ll.ffn.forward(state.ffn, x)

    return x
