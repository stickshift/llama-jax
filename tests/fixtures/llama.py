from jax import Array, dlpack
from jax import numpy as jnp
import numpy as np
import pytest
import torch

import llama_jax as ll
from llama_jax.benchmarks.llama_models import Transformer
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rope import Rope
from llama_jax.tokenizer import Tokenizer

__all__ = [
    "attention_norm0",
    "attention_output",
    "bs",
    "config",
    "ffn_output",
    "logits",
    "mask",
    "n",
    "params",
    "position_mask",
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


@pytest.fixture(scope="session")
def config() -> ModelConfig:
    """Llama3.2-3B model config."""

    config = ll.checkpoint.load_config("Llama3.2-3B")

    return config


@pytest.fixture(scope="session")
def params(config: ModelConfig) -> ModelParameters:
    """Llama3.2-3B model parameters."""
    return ll.checkpoint.load_parameters(config)


@pytest.fixture(scope="session")
def tokenizer(config: ModelConfig) -> Tokenizer:
    """Llama3 tokenizer."""
    return ll.checkpoint.load_tokenizer(config)


@pytest.fixture(scope="session")
def bs() -> int:
    """Batch size."""
    return len(_prompts)


@pytest.fixture(scope="session")
def n() -> int:
    """Sequence length."""

    # Sanity check
    assert all(len(x) == len(_token_ids[0]) for x in _token_ids)

    return len(_token_ids[0])


@pytest.fixture(scope="session")
def token_ids(config: ModelConfig, bs: int, n: int) -> Array:
    """Sample token ids."""

    # Sanity check
    assert all(len(x) == len(_token_ids[0]) for x in _token_ids)

    x = jnp.array(_token_ids)

    # Sanity check
    assert x.shape == (bs, n)

    # Token ids are always int32
    assert x.dtype == jnp.int32

    return x


@pytest.fixture(scope="session")
def position_mask(config: ModelConfig, bs: int, n: int) -> Array:
    """Sample position mask."""

    return ll.position_mask.create(jnp.ones((bs, n), dtype=jnp.int32), max_tokens=config.max_tokens)


@pytest.fixture(scope="session")
def token_embeddings(config: ModelConfig, bs: int, n: int, torch_device, reference_model: Transformer) -> Array:
    """Sample token embeddings."""

    # Load token ids into torch
    token_ids = torch.tensor(_token_ids, device=torch_device)

    # Map token_ids to embeddings using transformers as reference implementation
    x = reference_model.tok_embeddings(token_ids)

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Convert dtype
    x = x.astype(config.dtype)

    # Sanity check
    assert x.shape == (bs, n, config.d_model)

    return x


@pytest.fixture(scope="session")
def attention_norm0(
    config: ModelConfig,
    bs: int,
    n: int,
    token_embeddings: Array,
    torch_device,
    reference_model: Transformer,
) -> Array:
    """Sample attention outputs for layer 0."""
    layer = reference_model.layers[0]

    # Load embeddings into torch
    x = torch.tensor(np.array(token_embeddings.astype(jnp.float32)), device=torch_device)

    # Normalize
    x = layer.attention_norm(x)

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Convert dtype
    x = x.astype(config.dtype)

    # Sanity check
    assert x.shape == (bs, n, config.d_model)

    return x


@pytest.fixture(scope="session")
def attention_output(
    config: ModelConfig,
    bs: int,
    n: int,
    mask: Array,
    torch_device,
    reference_model: Transformer,
    token_embeddings: Array,
) -> Array:
    """Sample attention outputs for first layer."""

    layer = reference_model.layers[0]

    # Load embeddings into torch
    x = torch.tensor(np.array(token_embeddings.astype(jnp.float32)), device=torch_device)

    # Preserve residuals
    residuals = x

    # Normalize
    x = layer.attention_norm(x)

    # Generate (n, n) mask bias term
    m = torch.tensor(np.array(mask[0][:n, :n].astype(jnp.float32)), device=torch_device)

    # Attention
    freqs_cis = reference_model.freqs_cis[:n]
    x = layer.attention(x, start_pos=0, freqs_cis=freqs_cis, mask=m)

    # Merge residuals
    x = residuals + x

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Convert dtype
    x = x.astype(config.dtype)

    # Sanity check
    assert x.shape == (bs, n, config.d_model)

    return x


@pytest.fixture(scope="session")
def ffn_output(
    config: ModelConfig,
    bs: int,
    n: int,
    torch_device,
    reference_model: Transformer,
    attention_output: Array,
) -> Array:
    """Sample ffn outputs for layer 0."""
    layer = reference_model.layers[0]

    # Load attention values into torch
    x = torch.tensor(np.array(attention_output.astype(jnp.float32)), device=torch_device)

    # Preserve residuals
    residuals = x

    # Normalize
    x = layer.ffn_norm(x)

    # FFN
    x = layer.feed_forward(x)

    # Merge residuals
    x = residuals + x

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Convert dtype
    x = x.astype(config.dtype)

    # Sanity check
    assert x.shape == (bs, n, config.d_model)

    return x


@pytest.fixture(scope="session")
def logits(config: ModelConfig, bs: int, n: int, torch_device, reference_model: Transformer) -> Array:
    """Sample output logits."""

    # Load token ids into torch
    x = torch.tensor(_token_ids, device=torch_device)

    # Map token_ids to logits using reference model
    x = reference_model(x)

    # Convert logits from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Convert dtype
    x = x.astype(config.dtype)

    # Sanity check
    assert x.shape == (bs, n, config.vocab_size)

    return x


@pytest.fixture(scope="session")
def rope(config: ModelConfig, n: int) -> Rope:
    """RoPE matrices."""

    return ll.rope.create(config)


@pytest.fixture(scope="session")
def mask(config: ModelConfig, position_mask: Array) -> Array:
    """Attention mask."""

    return ll.attention.attention_mask(config, position_mask)
