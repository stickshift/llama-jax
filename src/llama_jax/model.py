"""Llama Model"""

from collections.abc import Mapping, Sequence, Set
from enum import Enum
import json
from pathlib import Path
import pickle
from typing import NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.dtypes import bfloat16
from jax.typing import ArrayLike, DTypeLike

__all__ = [
    "TrainingLevel",
    "ModelConfig",
    "load_config",
    "Model",
    "load_model",
    "RMSNorm",
    "rms",
    "Attention",
    "FFN",
    "Layer",
    "Head",
]


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


class TrainingLevel(str, Enum):
    PRETRAINED = "pretrained"
    INSTRUCT = "instruct"


class ModelConfig(NamedTuple):
    """Llama3 model config."""

    checkpoint_path: Path

    vocab_size: int

    d_model: int

    d_head: int

    d_ffn: int

    n_layers: int

    n_heads: int

    n_kv_heads: int

    rms_norm_eps: float

    rope_theta: float

    dtype: DTypeLike

    training: TrainingLevel


def load_config(checkpoint_name: str, **kwargs) -> ModelConfig:
    """Load Llama3 config from checkpoint params.json."""
    # Build checkpoint_path
    checkpoints_path = Path("~/.llama/checkpoints").expanduser()
    checkpoint_path = checkpoints_path / checkpoint_name

    # Validate
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Load hyperparameters
    hparams_path = checkpoint_path / "params.json"
    hparams = json.loads(hparams_path.read_text())

    # Calculate d_ffn from 8/3 * d_model rounded to nearest multiple_of
    d_model = hparams["dim"]
    ffn_dim_multiplier = hparams["ffn_dim_multiplier"]
    multiple_of = hparams["multiple_of"]
    d_ffn = int(8 / 3 * d_model * ffn_dim_multiplier)
    d_ffn = multiple_of * ((d_ffn + multiple_of - 1) // multiple_of)

    data = {
        "checkpoint_path": checkpoint_path,
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
        "dtype": bfloat16,
        "training": TrainingLevel.INSTRUCT if checkpoint_name.endswith("-Instruct") else TrainingLevel.PRETRAINED,
    }

    # Override with kwargs
    data |= kwargs

    return ModelConfig(**data)


# ------------------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------------------


class RMSNorm(NamedTuple):
    """RMS Normalization state."""

    weight: Array

    eps: float


def rms_norm(state: RMSNorm, x: ArrayLike) -> Array:
    """Normalize x using RMS Normalization.

    See https://doi.org/10.48550/arXiv.1910.07467
    """
    return state.weight * x / jnp.sqrt(jnp.mean(x**2) + state.eps)


# ------------------------------------------------------------------------------
# Attention
# ------------------------------------------------------------------------------

class Attention(NamedTuple):
    """Attention state."""

    n_heads: int

    n_kv_heads: int

    d_head: int

    norm: RMSNorm

    queries: Array

    keys: Array

    values: Array

    output: Array


# ------------------------------------------------------------------------------
# FFN
# ------------------------------------------------------------------------------

class FFN(NamedTuple):
    """Feedforward Network state."""

    norm: RMSNorm

    input: Array

    gate: Array

    output: Array


# ------------------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------------------

class Layer(NamedTuple):
    """Decoder layer state."""

    attention: Attention

    ffn: FFN


# ------------------------------------------------------------------------------
# Head
# ------------------------------------------------------------------------------

class Head(NamedTuple):
    """Head state."""

    norm: RMSNorm

    output: Array

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class Model(NamedTuple):
    """Model state."""

    embeddings: Array

    layers: Sequence[Layer]

    head: Head


def load_model(config: ModelConfig) -> Model:
    """Load model state from checkpoint."""

    # Validate
    input_path = config.checkpoint_path / "consolidated.00.jax"
    if not input_path.is_file():
        raise ValueError(
            f"Checkpoint {config.checkpoint_path.name} has not been converted to JAX format. See llama-jax CLI."
        )

    # Load state from checkpoint
    checkpoint_params = pickle.loads(input_path.read_bytes())  # noqa

    # Remap Meta's parameter names
    params = {}

    # Inject prefix for multimodal checkpoints
    prefix = "text_model." if "text_model.tok_embeddings.weight" in checkpoint_params else ""

    # Validate dtype
    assert checkpoint_params[f"{prefix}tok_embeddings.weight"].dtype == config.dtype

    # Embeddings
    embeddings = checkpoint_params[f"{prefix}tok_embeddings.weight"]

    # Layers
    layers = ()
    for layer_id in range(config.n_layers):
        attention = Attention(
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            norm=RMSNorm(
                weight=checkpoint_params[f"{prefix}layers.{layer_id}.attention_norm.weight"],
                eps=config.rms_norm_eps,
            ),
            queries=checkpoint_params[f"{prefix}layers.{layer_id}.attention.wq.weight"],
            keys=checkpoint_params[f"{prefix}layers.{layer_id}.attention.wk.weight"],
            values=checkpoint_params[f"{prefix}layers.{layer_id}.attention.wv.weight"],
            output=checkpoint_params[f"{prefix}layers.{layer_id}.attention.wo.weight"],
        )

        ffn = FFN(
            norm=RMSNorm(
                weight=checkpoint_params[f"{prefix}layers.{layer_id}.ffn_norm.weight"],
                eps=config.rms_norm_eps,
            ),
            input=checkpoint_params[f"{prefix}layers.{layer_id}.feed_forward.w3.weight"],
            gate=checkpoint_params[f"{prefix}layers.{layer_id}.feed_forward.w1.weight"],
            output=checkpoint_params[f"{prefix}layers.{layer_id}.feed_forward.w2.weight"],
        )

        layers += (Layer(attention=attention, ffn=ffn),)

    # Head
    head = Head(
        norm=RMSNorm(
            weight=checkpoint_params[f"{prefix}norm.weight"],
            eps=config.rms_norm_eps,
        ),
        output=checkpoint_params[f"{prefix}output.weight"],
    )

    return Model(embeddings=embeddings, layers=layers, head=head)
