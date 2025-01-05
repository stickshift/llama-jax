from jax import Array

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.kv_cache import LayerKVCache

from tests.fixtures.jax_fixtures import assert_similar_arrays


def test_factory():
    #
    # Givens
    #

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    #
    # Whens
    #

    # I create Attention for layers.0.attention
    attention = ll.attention.create(config, params, "layers.0.attention")

    #
    # Thens
    #

    # attention should be populated
    assert attention.queries.shape == (config.d_model, config.n_heads * config.d_head)
    assert attention.queries.dtype == config.dtype

    assert attention.keys.shape == (config.d_model, config.n_kv_heads * config.d_head)
    assert attention.keys.dtype == config.dtype

    assert attention.values.shape == (config.d_model, config.n_kv_heads * config.d_head)
    assert attention.values.dtype == config.dtype

    assert attention.output.shape == (config.d_model, config.d_model)
    assert attention.output.dtype == config.dtype


def test_attention_heads(config: ModelConfig, bs: int, n: int, token_embeddings: Array):
    #
    # Givens
    #

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I split attention heads
    y = ll.attention.split_heads(x, config.n_heads)

    #
    # Thens
    #

    # shape should be bs x n_heads x n x d_head
    assert y.shape == (bs, config.n_heads, n, config.d_head)

    #
    # Whens
    #

    # I combine attention heads
    y = ll.attention.combine_heads(y)

    #
    # Thens
    #

    # shape should be restored
    assert y.shape == x.shape

    # y should equal x
    assert (y == x).all()


def test_attention_mask(config: ModelConfig, n: int):
    #
    # Whens
    #

    # I generate mask
    m = ll.attention.attention_mask(config, n)

    #
    # Thens
    #

    # m.shape should be (n, n)
    assert m.shape == (n, n)


def test_forward(
    config: ModelConfig,
    params: ModelParameters,
    token_embeddings: Array,
    attention_output: Array,
):
    #
    # Givens
    #

    # I created Attention for layers.0.attention
    attention = ll.attention.create(config, params, "layers.0.attention")

    # I created a key/value cache
    kv_cache = LayerKVCache()

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ attention
    x, kv_cache = ll.attention.forward(config, attention, x, kv_cache)

    #
    # Thens
    #

    # x should match expected output
    assert_similar_arrays(x, attention_output)
