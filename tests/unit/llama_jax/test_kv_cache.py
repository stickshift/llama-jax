import jax
from jax import Array, random

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters


def test_apply(config: ModelConfig, params: ModelParameters, bs: int, n: int, key: Array, token_embeddings: Array):
    #
    # Givens
    #

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config, bs)

    # I loaded parameters for layer 0
    keys = params["layers.0.attention.wk.weight"].transpose()
    values = params["layers.0.attention.wv.weight"].transpose()
    layer_kvc = kv_cache[0]

    #
    # Whens
    #

    # I record shape of original layer cache
    shape0 = (layer_kvc.key_cache.shape, layer_kvc.value_cache.shape)

    # I initialize x w/ token embeddings
    x = token_embeddings

    # I generate keys and values for x
    k0 = ll.attention.split_heads(x @ keys, config.n_kv_heads)
    v0 = ll.attention.split_heads(x @ values, config.n_kv_heads)

    #
    # Thens
    #

    # k0, v0 should have length n
    assert k0.shape[-2] == n
    assert v0.shape[-2] == n

    #
    # Whens
    #

    # I apply layer cache
    layer_kvc, k1, v1 = ll.kv_cache.apply(layer_kvc, keys=k0, values=v0)

    # I record shape of updated layer cache
    shape1 = (layer_kvc.key_cache.shape, layer_kvc.value_cache.shape)

    #
    # Thens
    #

    # k1, v1 should equal k0, v0 since the cache was previously empty
    assert (k1 == k0).all()
    assert (v1 == v0).all()

    # Layer cache should not have changed shape
    assert shape1 == shape0

    #
    # Whens
    #

    # I generate embeddings for next token
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, 1, config.d_model), dtype=config.dtype)

    # I generate keys and values for x
    k2 = ll.attention.split_heads(x @ keys, config.n_kv_heads)
    v2 = ll.attention.split_heads(x @ values, config.n_kv_heads)

    #
    # Thens
    #

    # k2, v2 should have length 1
    assert k2.shape[-2] == 1
    assert v2.shape[-2] == 1

    #
    # Whens
    #

    # I apply layer cache
    layer_kvc, k3, v3 = ll.kv_cache.apply(layer_kvc, keys=k2, values=v2)

    # I record shape of updated layer cache
    shape2 = (layer_kvc.key_cache.shape, layer_kvc.value_cache.shape)

    #
    # Thens
    #

    # k3, v3 should have length n + 1
    assert k3.shape[-2] == n + 1
    assert v3.shape[-2] == n + 1

    # Layer cache should not have changed shape
    assert shape2 == shape0
