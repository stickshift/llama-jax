"""Llama Model."""

from collections.abc import Sequence, Mapping
from enum import Enum
import json
from pathlib import Path
import pickle
from typing import NamedTuple

from jax import Array
from jax.dtypes import bfloat16
from jax.typing import DTypeLike

from .attention import Attention
from .ffn import FFN
from .head import Head
from .layer import Layer
from .rms_norm import RMSNorm
from .tokenizer import Tokenizer

__all__ = [
    "Model",
    "ModelConfig",
    "TrainingLevel",
    "load_config",
    "load_parameters",
    "load_model",
    "load_tokenizer",
]


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


class TrainingLevel(str, Enum):
    """Llama3 training level."""

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
# Parameters
# ------------------------------------------------------------------------------

ModelParameters = Mapping[str, Array]

def load_parameters(config: ModelConfig) -> ModelParameters:
    # Validate
    input_path = config.checkpoint_path / "consolidated.00.jax"
    if not input_path.is_file():
        raise ValueError(
            f"Checkpoint {config.checkpoint_path.name} has not been converted to JAX format. See llama-jax CLI."
        )

    # Load state from checkpoint
    params = pickle.loads(input_path.read_bytes())  # noqa

    return params


# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------


def load_tokenizer(config: ModelConfig) -> Tokenizer:
    """Load tokenizer from checkpoint."""
    # Load tiktoken model
    return Tokenizer(str(config.checkpoint_path / "tokenizer.model"))


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

    # Load state from checkpoint
    params = load_parameters(config)

    # Inject prefix for multimodal checkpoints
    prefix = "text_model." if "text_model.tok_embeddings.weight" in params else ""

    # Validate dtype
    assert params[f"{prefix}tok_embeddings.weight"].dtype == config.dtype

    # Embeddings
    embeddings = params[f"{prefix}tok_embeddings.weight"]

    # Layers
    layers = ()
    for layer_id in range(config.n_layers):
        attention = Attention(
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            norm=RMSNorm(
                weight=params[f"{prefix}layers.{layer_id}.attention_norm.weight"],
                eps=config.rms_norm_eps,
            ),
            queries=params[f"{prefix}layers.{layer_id}.attention.wq.weight"],
            keys=params[f"{prefix}layers.{layer_id}.attention.wk.weight"],
            values=params[f"{prefix}layers.{layer_id}.attention.wv.weight"],
            output=params[f"{prefix}layers.{layer_id}.attention.wo.weight"],
        )

        ffn = FFN(
            norm=RMSNorm(
                weight=params[f"{prefix}layers.{layer_id}.ffn_norm.weight"],
                eps=config.rms_norm_eps,
            ),
            input=params[f"{prefix}layers.{layer_id}.feed_forward.w3.weight"],
            gate=params[f"{prefix}layers.{layer_id}.feed_forward.w1.weight"],
            output=params[f"{prefix}layers.{layer_id}.feed_forward.w2.weight"],
        )

        layers += (Layer(attention=attention, ffn=ffn),)

    # Head
    head = Head(
        norm=RMSNorm(
            weight=params[f"{prefix}norm.weight"],
            eps=config.rms_norm_eps,
        ),
        output=params[f"{prefix}output.weight"],
    )

    return Model(embeddings=embeddings, layers=layers, head=head)
