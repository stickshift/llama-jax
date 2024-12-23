from jax import random
import pytest

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.ffn import FFN
from llama_jax.kv_cache import LayerKVCache


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


def test_forward(bs: int, n: int):
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

    # I created rope and attention mask
    rope = ll.rope.create(config, n)
    mask = ll.attention.attention_mask(n, config.dtype)

    # I created a key/value cache
    kv_cache = LayerKVCache()

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, n, config.d_model))

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
