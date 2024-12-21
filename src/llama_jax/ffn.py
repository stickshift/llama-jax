"""Feedforward Network."""

from typing import NamedTuple

from jax import Array
from jax.nn import silu
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm

__all__ = [
    "FFN",
    "create",
    "forward",
]


class FFN(NamedTuple):
    """Feedforward Network state."""

    norm: RMSNorm

    input: Array

    gate: Array

    output: Array


def create(config: ModelConfig, params: ModelParameters, path: str) -> FFN:
    """Load Llama3 FFN."""
    parent_path = path.rsplit(".", 1)[0]

    # Note we transpose kernels so we don't need to during forward pass
    input = params[f"{path}.w3.weight"].transpose()
    gate = params[f"{path}.w1.weight"].transpose()
    output = params[f"{path}.w2.weight"].transpose()

    return FFN(
        norm=ll.rms_norm.create(config, params, f"{parent_path}.ffn_norm"),
        input=input,
        gate=gate,
        output=output,
    )


def forward(config: ModelConfig, state: FFN, x: ArrayLike) -> Array:
    """Transform x using feedforward network (FFN)."""

    # Sanity check
    assert x.ndim == 3

    # Save residuals
    residual = x

    # Normalize inputs
    x = ll.rms_norm.forward(config, state.norm, x)

    # Apply SwiGLU transform
    x = silu(x @ state.gate) * (x @ state.input)

    # Project outputs back to model space
    x = x @ state.output

    # Merge outputs with residuals
    x = residual + x

    return x
