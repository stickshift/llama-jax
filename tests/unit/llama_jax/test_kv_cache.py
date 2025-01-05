from jax import Array, random

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.kv_cache import LayerKVCache


def test_apply(config: ModelConfig, params: ModelParameters, bs: int, key: Array, token_embeddings: Array):
    #
    # Givens
    #

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    # I loaded parameters for layer 0
    keys = params["layers.0.attention.wk.weight"].transpose()
    values = params["layers.0.attention.wv.weight"].transpose()
    layer_kv_cache = kv_cache[0]

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I generate keys and values for initial sequence of embeddings
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

    # I generate random next token embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, 1, config.d_model))

    # I generate keys and values for next token
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
