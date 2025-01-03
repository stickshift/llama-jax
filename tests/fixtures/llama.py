from jax import Array, dlpack
from jax import numpy as jnp
import pytest
import torch
import numpy as np

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters, TOKEN_AXIS
from llama_jax.rope import Rope
from llama_jax.tokenizer import Tokenizer
from llama_jax.benchmarks.llama_models import Transformer

__all__ = [
    "bs",
    "config",
    "attention_norm0",
    "attention_0",
    "attention_n",
    "logits",
    "mask",
    "n",
    "params",
    "rope",
    "token_embeddings",
    "token_ids",
    "tokenizer",
    "ffn_0",
    "ffn_n",
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
    return ll.checkpoint.load_config("Llama3.2-3B")


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

    return x


@pytest.fixture(scope="session")
def token_embeddings(config: ModelConfig, bs: int, n: int, torch_device, reference_model: Transformer) -> Array:
    """Sample token embeddings."""

    # Load token ids into torch
    token_ids = torch.tensor(_token_ids, device=torch_device)

    # Map token_ids to embeddings using transformers as reference implementation
    x = reference_model.tok_embeddings(token_ids)

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

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
    x = torch.tensor(np.array(token_embeddings), device=torch_device)

    # Preserve residuals
    residuals = x

    # Normalize
    x = layer.attention_norm(x)

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

    return x


@pytest.fixture(scope="session")
def attention_0(
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
    x = torch.tensor(np.array(token_embeddings), device=torch_device)

    # Preserve residuals
    residuals = x

    # Normalize
    x = layer.attention_norm(x)

    # Attention
    freqs_cis = reference_model.freqs_cis[:n]
    mask = torch.tensor(np.array(mask), device=torch_device)
    x = layer.attention(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)

    # Merge residuals
    x = residuals + x

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

    return x


@pytest.fixture(scope="session")
def attention_n(
    config: ModelConfig,
    bs: int,
    n: int,
    mask: Array,
    torch_device,
    reference_model: Transformer,
    token_embeddings: Array,
) -> Array:
    """Sample attention outputs for last layer."""

    layer = reference_model.layers[0]

    # Load embeddings into torch
    x = torch.tensor(np.array(token_embeddings), device=torch_device)

    # Layers
    start_pos = 0
    freqs_cis = reference_model.freqs_cis[:n]
    mask = torch.tensor(np.array(mask), device=torch_device)
    for i, layer in enumerate(reference_model.layers):
        # Attention
        x = x + layer.attention(layer.attention_norm(x), start_pos, freqs_cis, mask)

        # Bail on last layer
        if i == config.n_layers - 1:
            break

        # FFN
        x = x + layer.feed_forward(layer.ffn_norm(x))

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

    return x


@pytest.fixture(scope="session")
def ffn_0(
    config: ModelConfig,
    bs: int,
    n: int,
    torch_device,
    reference_model: Transformer,
    attention_0: Array,
) -> Array:
    """Sample ffn outputs for layer 0."""
    layer = reference_model.layers[0]

    # Load attention values into torch
    x = torch.tensor(np.array(attention_0), device=torch_device)

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

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

    return x


@pytest.fixture(scope="session")
def ffn_n(
    config: ModelConfig,
    bs: int,
    n: int,
    mask: Array,
    torch_device,
    reference_model: Transformer,
    token_embeddings: Array,
) -> Array:
    """Sample ffn outputs for last layer."""

    layer = reference_model.layers[0]

    # Load embeddings into torch
    x = torch.tensor(np.array(token_embeddings), device=torch_device)

    # Layers
    start_pos = 0
    freqs_cis = reference_model.freqs_cis[:n]
    mask = torch.tensor(np.array(mask), device=torch_device)
    for i, layer in enumerate(reference_model.layers):
        # Attention
        x = x + layer.attention(layer.attention_norm(x), start_pos, freqs_cis, mask)

        # FFN
        x = x + layer.feed_forward(layer.ffn_norm(x))

    # Convert from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Sanity check
    assert x.shape == (bs, n, config.d_model)
    assert x.dtype == config.dtype

    return x


@pytest.fixture(scope="session")
def logits(config: ModelConfig, bs: int, torch_device, reference_model: Transformer) -> Array:
    """Sample output logits."""

    # Load token ids into torch
    x = torch.tensor(_token_ids, device=torch_device)

    # Map token_ids to logits using reference model
    x = reference_model(x)

    # Convert logits from torch to jax
    x = dlpack.from_dlpack(x.cpu())

    # Only keep logits for last embedding
    x = x[:, -1]

    # Sanity check
    assert x.shape == (bs, config.vocab_size)

    return x


@pytest.fixture(scope="session")
def rope(config: ModelConfig, n: int) -> Rope:
    """RoPE matrices."""

    return ll.rope.create(config, n)


@pytest.fixture(scope="session")
def mask(config: ModelConfig, n: int) -> Array:
    """Causal attention mask."""

    return ll.attention.attention_mask(n, config.dtype)
