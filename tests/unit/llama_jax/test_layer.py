from jax import Array

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.ffn import FFN
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


def test_forward(
    config: ModelConfig,
    params: ModelParameters,
    rope: Rope,
    mask: Array,
    token_embeddings: Array,
):
    #
    # Givens
    #

    # I created Layer for layers.0
    layer = ll.layer.create(config, params, "layers.0")

    # I created a key/value cache
    layer_kvc = ll.kvc.create(config)[0]

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ layer
    y, layer_kvc = ll.layer.forward(config, layer, rope, mask, x, layer_kvc)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
