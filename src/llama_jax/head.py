"""Head."""

from typing import NamedTuple

from jax import Array
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
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


def forward(config: ModelConfig, state: Head, x: Array, position_mask: Array) -> Array:
    """Transform embeddings into token logits."""
    bs, n, d_model = x.shape[0], x.shape[1], config.d_model

    # Select first n values of position_mask
    indices = jnp.broadcast_to(jnp.arange(n), (bs, n))
    mask = jnp.take_along_axis(position_mask, indices, axis=-1)

    # Normalize inputs
    x = ll.rms_norm.forward(config, state.norm, x)

    # Use last "real" embedding to represent the entire sequence.

    # Start with fixed lengths for all sequences
    fixed_lengths = jnp.full(bs, fill_value=n)

    # Calculate custom lengths per sequence w/o padding
    padded_lengths = jnp.sum(mask, axis=-1)

    # Use padded lengths if the last position in any sequence is padded
    condition = jnp.any(jnp.take(mask, -1, axis=-1) == 0)
    lengths = jnp.where(condition, padded_lengths, fixed_lengths)

    # Select the lengths-1 embedding for each batch
    indices = jnp.reshape(lengths - 1, (bs, 1, 1))
    x = jnp.take_along_axis(x, indices, axis=-2)
    x = jnp.reshape(x, (bs, d_model))

    # Project outputs to token space
    x = x @ state.output

    return x
