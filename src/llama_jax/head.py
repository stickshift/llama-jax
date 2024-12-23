"""Head."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "Head",
    "create",
    "forward",
]


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

_TOKEN_AXIS = -2


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
    #   Note: If x is batched, we need to swap token dimension to front before using x[-1]
    #
    swap = x.ndim > 2
    if swap:
        x = jnp.swapaxes(x, 0, _TOKEN_AXIS)

    x = x[-1]

    if swap:
        x = jnp.swapaxes(x, 0, _TOKEN_AXIS)

    # Project outputs to token space
    x = x @ state.output

    return x
