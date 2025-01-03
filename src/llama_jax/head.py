"""Head."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters, TOKEN_AXIS
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "Head",
    "create",
    "forward",
]


# ------------------------------------------------------------------------------
# Head
# ------------------------------------------------------------------------------


class Head(NamedTuple):
    """Head state."""

    norm: RMSNorm

    output: Array


def create(config: ModelConfig, params: ModelParameters) -> Head:
    """Load Llama3 Head."""
    # Note we transpose kernels so we don't need to during forward pass
    output = params["output.weight"].transpose()

    return Head(
        norm=ll.rms_norm.create(config, params, "norm"),
        output=output,
    )


def forward(config: ModelConfig, state: Head, x: ArrayLike) -> Array:
    """Transform embeddings into token logits."""
    # Normalize inputs
    x = ll.rms_norm.forward(config, state.norm, x)

    # Use last embedding to represent the entire sequence.
    x = jnp.swapaxes(x, 0, TOKEN_AXIS)[-1]

    # Project outputs to token space
    x = x @ state.output

    return x
