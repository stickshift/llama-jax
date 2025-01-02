from jax import Array, dlpack
from jax import numpy as jnp
import pytest
import torch

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rope import Rope
from llama_jax.tokenizer import Tokenizer

__all__ = [
    "bs",
    "config",
    "mask",
    "n",
    "params",
    "rope",
    "token_embeddings",
    "token_ids",
    "tokenizer",
]

_prompts = [
    "I like traveling by train because it is a relaxing way to get from one place to another.",
    "I like to sit back and enjoy the scenery. I like to read a book or listen.",
]

_token_ids = [
    [128000, 40, 1093, 21646, 555, 5542, 1606, 433, 374, 264, 34948, 1648, 311, 636, 505, 832, 2035, 311, 2500, 13],
    [128000, 40, 1093, 311, 2503, 1203, 323, 4774, 279, 51331, 13, 358, 1093, 311, 1373, 264, 2363, 477, 9020, 13],
]


@pytest.fixture
def config() -> ModelConfig:
    """Llama3.2-3B model config."""
    return ll.checkpoint.load_config("Llama3.2-3B")


@pytest.fixture
def params(config: ModelConfig) -> ModelParameters:
    """Llama3.2-3B model parameters."""
    return ll.checkpoint.load_parameters(config)


@pytest.fixture
def tokenizer(config: ModelConfig) -> Tokenizer:
    """Llama3 tokenizer."""
    return ll.checkpoint.load_tokenizer(config)


@pytest.fixture
def bs() -> int:
    """Batch size."""
    return len(_prompts)


@pytest.fixture
def n() -> int:
    """Sequence length."""

    # Sanity check
    assert all(len(x) == len(_token_ids[0]) for x in _token_ids)

    return len(_token_ids[0])


@pytest.fixture
def token_ids() -> Array:
    """Sample token ids."""

    # Sanity check
    assert all(len(x) == len(_token_ids[0]) for x in _token_ids)

    return jnp.array(_token_ids)


@pytest.fixture(scope="session")
def token_embeddings(torch_device, transformers_model) -> Array:
    """Sample token embeddings."""

    # Map token_ids to embeddings using transformers as reference implementation
    token_ids = torch.tensor(_token_ids, device=torch_device)
    x = transformers_model.model.embed_tokens(token_ids)

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    return x


@pytest.fixture
def rope(config: ModelConfig, n: int) -> Rope:
    """RoPE matrices."""

    return ll.rope.create(config, n)


@pytest.fixture
def mask(config: ModelConfig, n: int) -> Array:
    """Causal attention mask."""

    return ll.attention.attention_mask(n, config.dtype)
