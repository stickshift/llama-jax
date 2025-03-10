from jax import Array
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rope import Rope

from tests.fixtures.jax_fixtures import assert_similar_arrays


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Givens
    #

    # I overrode config dtype
    config = config._replace(dtype=jnp.int32)

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


def test_forward(
    config: ModelConfig,
    params: ModelParameters,
    rope: Rope,
    mask: Array,
    token_embeddings: Array,
    attention_output: Array,
):
    #
    # Givens
    #

    # I created Attention for layers.0.attention
    attention = ll.attention.create(config, params, "layers.0.attention")

    # I created a key/value cache
    layer_kvc = ll.kvc.create(config)[0]

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ attention
    y, layer_kvc = ll.attention.forward(config, attention, rope, mask, x, layer_kvc)

    #
    # Thens
    #

    # y should match expected output
    assert_similar_arrays(y, attention_output)
