from jax import numpy as jnp

import llama_jax as ll


def test_factory():
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

    # I create Rope
    rope = ll.rope.create(config, n)

    #
    # Thens
    #

    # rotation matrices should have shape (n, d_head)
    assert rope.cos.shape == (n, config.d_head)
    assert rope.sin.shape == (n, config.d_head)


def test_swap():
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
    y = ll.rope.swap(x)

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
