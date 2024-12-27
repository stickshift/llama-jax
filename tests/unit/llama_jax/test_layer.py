from jax import Array

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.ffn import FFN
from llama_jax.kv_cache import LayerKVCache
from llama_jax.rope import Rope


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Whens
    #

    # I create Layer for layers.0
    layer = ll.layer.create(config, params, "layers.0")

    #
    # Thens
    #

    # layer should be populated
    assert isinstance(layer.attention, Attention)
    assert isinstance(layer.ffn, FFN)


def test_forward(config: ModelConfig, params: ModelParameters, token_embeddings: Array, rope: Rope, mask: Array):
    #
    # Givens
    #

    # I created Layer for layers.0
    layer = ll.layer.create(config, params, "layers.0")

    # I created a key/value cache
    kv_cache = LayerKVCache()

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ layer
    y, kv_cache = ll.layer.forward(config, layer, rope, mask, kv_cache, x)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape

    # kv_cache should be populated
    assert kv_cache.keys is not None
    assert kv_cache.values is not None
