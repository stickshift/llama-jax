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


def forward(state: Head, x: ArrayLike) -> Array:
    """Transform embeddings into token logits."""
    # Normalize inputs
    x = ll.rms_norm.forward(state.norm, x)

    # Use last embedding to represent the entire sequence
    x = x[-1]

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
) -> Array:
    """Select next token using temperature, top_p, and top_k sampling."""
    # Defaults
    key = default_arg(key, default_factory=random.key)
    temperature = default_arg(temperature, 0.6)
    top_k = default_arg(top_k, 50)
    top_p = default_arg(top_p, 0.9)

    # Temperature
    # -----------

    # If temperature is 0, return the top token
    if temperature == 0:
        return jnp.argmax(logits, axis=-1)

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
    key, subkey = random.split(key)
    sampled_index = random.choice(subkey, jnp.arange(probs.shape[0]), p=probs)

    # Convert sampled_index to original logits
    token_id = indices[sampled_index]

    return token_id
