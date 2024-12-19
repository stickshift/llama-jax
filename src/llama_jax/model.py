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
]


class Model(NamedTuple):
    """Model state."""

    embeddings: Array

    layers: Sequence[Layer]

    head: Head
#
#
# def load_model(config: ModelConfig) -> Model:
#     """Load model state from checkpoint."""
#
#     # Load state from checkpoint
#     params = load_parameters(config)
#
#     # Inject prefix for multimodal checkpoints
#     prefix = "text_model." if "text_model.tok_embeddings.weight" in params else ""
#
#     # Validate dtype
#     assert params[f"{prefix}tok_embeddings.weight"].dtype == config.dtype
#
#     # Embeddings
#     embeddings = params[f"{prefix}tok_embeddings.weight"]
#
#     # Layers
#     layers = ()
#     for layer_id in range(config.n_layers):
#         attention = Attention(
#             n_heads=config.n_heads,
#             n_kv_heads=config.n_kv_heads,
#             d_head=config.d_head,
#             norm=RMSNorm(
#                 weight=params[f"{prefix}layers.{layer_id}.attention_norm.weight"],
#                 eps=config.rms_norm_eps,
#             ),
#             queries=params[f"{prefix}layers.{layer_id}.attention.wq.weight"],
#             keys=params[f"{prefix}layers.{layer_id}.attention.wk.weight"],
#             values=params[f"{prefix}layers.{layer_id}.attention.wv.weight"],
#             output=params[f"{prefix}layers.{layer_id}.attention.wo.weight"],
#         )
#
#         ffn = FFN(
#             norm=RMSNorm(
#                 weight=params[f"{prefix}layers.{layer_id}.ffn_norm.weight"],
#                 eps=config.rms_norm_eps,
#             ),
#             input=params[f"{prefix}layers.{layer_id}.feed_forward.w3.weight"],
#             gate=params[f"{prefix}layers.{layer_id}.feed_forward.w1.weight"],
#             output=params[f"{prefix}layers.{layer_id}.feed_forward.w2.weight"],
#         )
#
#         layers += (Layer(attention=attention, ffn=ffn),)
#
#     # Head
#     head = Head(
#         norm=RMSNorm(
#             weight=params[f"{prefix}norm.weight"],
#             eps=config.rms_norm_eps,
#         ),
#         output=params[f"{prefix}output.weight"],
#     )
#
#     return Model(embeddings=embeddings, layers=layers, head=head)
