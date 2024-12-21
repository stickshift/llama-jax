"""Head."""

from typing import NamedTuple

from jax import Array, random
from jax import numpy as jnp
from jax.nn import softmax
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm
from llama_jax.tools import default_arg

__all__ = [
    "Head",
    "create",
    "forward",
    "sample_token",
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


def sample_token(
    logits: ArrayLike,
    *,
    key: ArrayLike | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[Array, Array]:
    """Select next token using temperature, top_p, and top_k sampling."""
    # Defaults
    key = default_arg(key, default_factory=random.key)
    temperature = default_arg(temperature, 0.6)
    top_k = default_arg(top_k, 50)
    top_p = default_arg(top_p, 0.9)

    # Split key
    key, subkey = random.split(key)

    # Temperature
    # -----------

    # If temperature is 0, return the top token
    if temperature == 0:
        return jnp.argmax(logits, axis=-1), key

    # Apply temperature
    logits = logits / temperature

    # Ranking
    # -------

    # Convert logits to probabilities
    probs = softmax(logits, axis=-1)

    # Sort probabilities in descending order, maintaining original indices
    indices = jnp.argsort(probs, descending=True)
    probs = probs[indices]

    # Top K
    # -----

    # Retain top k tokens
    probs = probs[:top_k]

    # Top P
    # -----

    # Find cutoff where cumulative probability exceeds top_p
    cumulative_mask = probs.cumsum() > top_p
    threshold_index = jnp.argmax(cumulative_mask).item()

    # Only apply threshold if top_p was exceeded
    if cumulative_mask.any():
        probs = probs[: threshold_index + 1]

    # Random Selection
    # ----------------

    # Sample from remaining tokens weighted by probability
    sampled_index = random.choice(subkey, jnp.arange(probs.shape[0]), p=probs)

    # Convert sampled_index to original logits
    token_id = indices[sampled_index]

    return token_id, key
