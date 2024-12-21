"""Decoder layer."""

from typing import NamedTuple

from jax import Array
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.ffn import FFN
from llama_jax.rope import Rope

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


def forward(
    config: ModelConfig,
    state: Layer,
    rope: Rope,
    mask: ArrayLike,
    x: ArrayLike,
) -> Array:
    """Transform x using attention and feedforward network."""
    # Attention
    x = ll.attention.forward(config, state.attention, rope, mask, x)

    # FFN
    x = ll.ffn.forward(config, state.ffn, x)

    return x
