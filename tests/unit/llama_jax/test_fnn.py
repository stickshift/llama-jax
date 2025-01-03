from jax import Array

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.kv_cache import MutableKVCache
from llama_jax.rope import Rope

from tests.fixtures.jax_fixtures import assert_similar_arrays


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Whens
    #

    # I create FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    #
    # Thens
    #

    # ffn should be populated
    assert ffn.input.shape == (config.d_model, config.d_ffn)
    assert ffn.gate.shape == (config.d_model, config.d_ffn)
    assert ffn.output.shape == (config.d_ffn, config.d_model)


def test_forward_0(
    config: ModelConfig,
    params: ModelParameters,
    attention_0: Array,
    ffn_0: Array,
):
    #
    # Givens
    #

    # I created FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    # Sample attention output
    x = attention_0

    #
    # Whens
    #

    # I transform x w/ ffn
    x = ll.ffn.forward(config, ffn, x)

    #
    # Thens
    #

    # x should match expected output
    assert_similar_arrays(x, ffn_0)


def test_forward_n(
    config: ModelConfig,
    params: ModelParameters,
    rope: Rope,
    mask: Array,
    bs: int,
    n: int,
    token_embeddings: Array,
    ffn_n: Array,
):
    #
    # Givens
    #

    # I created model
    model = ll.model.create(config, params)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)
    kv_cache = MutableKVCache(kv_cache)

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ n layers
    for i, layer in enumerate(model.layers):
        # Attention
        x, kv_cache[i] = ll.attention.forward(config, layer.attention, rope, mask, kv_cache[i], x)

        # FFN
        x = ll.ffn.forward(config, layer.ffn, x)

    #
    # Thens
    #

    # x should match expected output
    assert_similar_arrays(x, ffn_n)
