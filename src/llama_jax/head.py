"""Head."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.nn import silu
from jax.typing import ArrayLike, DTypeLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "Head",
    "create",
    "forward",
]


class Head(NamedTuple):
    """Head state."""

    norm: RMSNorm

    output: Array


def create(config: ModelConfig, params: ModelParameters) -> Head:
    """Load Llama3 Head."""

    # Note we transpose kernels so we don't need to during forward pass
    output = params[f"output.weight"].transpose()

    return Head(
        norm=ll.rms_norm.create(config, params, f"norm"),
        output=output,
    )


def forward(state: Head, x: ArrayLike) -> Array:
    """Transform embeddings into token logits."""

    # Normalize inputs
    x = ll.rms_norm.forward(state.norm, x)

    # Use last embedding to represent the entire sequence
    x = x[-1]

    # Project outputs to token space
    x = x @ state.output

    return x
