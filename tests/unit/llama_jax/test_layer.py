from jax import random
import pytest

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.ffn import FFN


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

    # I create Layer for layers.0
    layer = ll.layer.create(config, params, "layers.0")

    #
    # Thens
    #

    # layer should be populated
    assert isinstance(layer.attention, Attention)
    assert isinstance(layer.ffn, FFN)


@pytest.mark.parametrize("input_shape", [(10, 3072), (2, 10, 3072)])
def test_forward(input_shape: tuple):
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created Layer for layers.0
    layer = ll.layer.create(config, params, "layers.0")

    # I generated rope rotation matrices and masked attention bias
    n = input_shape[-2]
    r_cos, r_sin = ll.attention.rope_frequencies(config, n)
    m = ll.attention.masked_attention_bias(n, config.dtype)

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, input_shape)

    #
    # Whens
    #

    # I transform x w/ layer
    y = ll.layer.forward(config, layer, x, r_cos, r_sin, m)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
