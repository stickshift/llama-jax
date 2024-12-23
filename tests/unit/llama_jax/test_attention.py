import jax
from jax import numpy as jnp
from jax import random
import pytest

import llama_jax as ll
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

    # I create Attention for layers.0.attention
    attention = ll.attention.create(config, params, "layers.0.attention")

    #
    # Thens
    #

    # attention should be populated
    assert attention.queries.shape == (config.d_model, config.n_heads * config.d_head)
    assert attention.keys.shape == (config.d_model, config.n_kv_heads * config.d_head)
    assert attention.values.shape == (config.d_model, config.n_kv_heads * config.d_head)
    assert attention.output.shape == (config.d_model, config.d_model)


def test_masked_attention_bias():
    #
    # Givens
    #

    # Dimensions
    n = 10

    #
    # Whens
    #

    # I generate masked attention bias term M
    m = ll.attention.attention_mask(n, jax.dtypes.bfloat16)

    #
    # Thens
    #

    # m is (n, n) array with zeros below the diagonal, -inf above the diagonal
    for i in range(n):
        for j in range(n):
            if j > i:
                assert m[i, j] == -jnp.inf
            else:
                assert m[i, j] == 0


def test_attention_heads(bs: int, n: int):
    #
    # Givens
    #

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I generated sample embeddings
    x = jnp.arange(bs * n * config.d_model).reshape(bs, n, config.d_model)

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


def test_forward(bs: int, n: int):
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created Attention for layers.0.attention
    attention = ll.attention.create(config, params, "layers.0.attention")

    # I created rope and attention mask
    rope = ll.rope.create(config, n)
    mask = ll.attention.attention_mask(n, config.dtype)

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (bs, n, config.d_model))

    # I created a key/value cache
    kv_cache = LayerKVCache()

    #
    # Whens
    #

    # I transform x w/ attention
    y, kv_cache = ll.attention.forward(config, attention, rope, mask, kv_cache, x)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
