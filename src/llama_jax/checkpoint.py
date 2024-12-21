"""Llama Checkpoint."""

from collections.abc import Mapping
from enum import Enum
import json
from pathlib import Path
import pickle
from typing import NamedTuple

from jax import Array
from jax.dtypes import bfloat16
from jax.typing import DTypeLike

from .tokenizer import Tokenizer

__all__ = [
    "ModelConfig",
    "ModelParameters",
    "TrainingLevel",
    "load_config",
    "load_parameters",
    "load_tokenizer",
    "HEAD_AXIS",
    "TOKEN_AXIS",
    "MODEL_AXIS",
]

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

HEAD_AXIS = -3
TOKEN_AXIS = -2
MODEL_AXIS = -1


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
    """Load Llama3 parameters from checkpoint consolidated.00.jax."""
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
