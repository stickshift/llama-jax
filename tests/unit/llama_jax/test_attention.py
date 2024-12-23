import jax
from jax import numpy as jnp
from jax import random

import llama_jax as ll


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


def test_rope_frequencies():
    #
    # Givens
    #

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # sequence length
    n = 10

    #
    # Whens
    #

    # I generate rope rotation matrices
    r_cos, r_sin = ll.attention.rope_frequencies(config, n)

    #
    # Thens
    #

    # rotation matrices should have shape (n, d_head)
    assert r_cos.shape == (n, config.d_head)
    assert r_sin.shape == (n, config.d_head)


def test_rope_swap():
    #
    # Givens
    #

    # Dimensions
    n = 10
    m = 20
    d = 4

    # I generate sample n x m x d data
    x = jnp.arange(n * m * d).reshape(n, m, d)

    #
    # Whens
    #

    # I swap x
    y = ll.attention.rope_swap(x)

    #
    # Thens
    #

    # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2] along last dimension
    for i in range(n):
        for j in range(m):
            assert y[i, j, 0] == -x[i, j, 1]
            assert y[i, j, 1] == x[i, j, 0]
            assert y[i, j, 2] == -x[i, j, 3]
            assert y[i, j, 3] == x[i, j, 2]


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
    m = ll.attention.masked_attention_bias(n, jax.dtypes.bfloat16)

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


def test_attention_heads():
    #
    # Givens
    #

    # Dimensions
    n = 10
    d_model = 128
    d_head = 32
    n_heads = d_model // d_head

    # I generated sample embeddings
    x = jnp.arange(n * d_model).reshape(n, d_model)

    #
    # Whens
    #

    # I split attention heads
    y = ll.attention.split_heads(x, n_heads)

    #
    # Thens
    #

    # shape should be n_heads x n x d_head
    assert y.shape == (n_heads, n, d_head)

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


def test_forward():
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

    # sequence length
    n = 10

    # I generated rope rotation matrices and masked attention bias
    r_cos, r_sin = ll.attention.rope_frequencies(config, n)
    m = ll.attention.masked_attention_bias(n, config.dtype)

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (n, config.d_model))

    #
    # Whens
    #

    # I transform x w/ attention
    y = ll.attention.forward(config, attention, x, r_cos, r_sin, m)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
