from jax import random

import llama_jax as ll
from llama_jax.kv_cache import LayerKVCache


def test_update(bs: int, n: int):
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    # I loaded parameters for layer 0
    keys = params["layers.0.attention.wk.weight"].transpose()
    values = params["layers.0.attention.wv.weight"].transpose()
    layer_kv_cache = kv_cache[0]

    #
    # Whens
    #

    # I generate keys and values for initial sequence of tokens
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, n, config.d_model))
    k0 = ll.attention.split_heads(x @ keys, config.n_kv_heads)
    v0 = ll.attention.split_heads(x @ values, config.n_kv_heads)

    # I apply cache
    k1 = ll.kv_cache.apply(layer_kv_cache.keys, k0)
    v1 = ll.kv_cache.apply(layer_kv_cache.values, v0)

    #
    # Thens
    #

    # k1, v1 should equal k0, v0
    assert (k1 == k0).all()
    assert (v1 == v0).all()

    #
    # Whens
    #

    # I update layer_kv_cache
    layer_kv_cache = LayerKVCache(keys=k1, values=v1)

    # I generate keys and values for next token
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, 1, config.d_model))
    k2 = ll.attention.split_heads(x @ keys, config.n_kv_heads)
    v2 = ll.attention.split_heads(x @ values, config.n_kv_heads)

    # I apply cache
    k3 = ll.kv_cache.apply(layer_kv_cache.keys, k2)
    v3 = ll.kv_cache.apply(layer_kv_cache.values, v2)

    #
    # Thens
    #

    # k3.n should be k0.n + 1
    assert k3.shape[-2] == k0.shape[-2] + 1

    # v3.n should be v0.n + 1
    assert v3.shape[-2] == v0.shape[-2] + 1
