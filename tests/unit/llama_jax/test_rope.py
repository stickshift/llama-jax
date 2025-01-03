from jax import numpy as jnp
import jax.dtypes

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig


def test_factory(config: ModelConfig, n: int):
    #
    # Whens
    #

    # I initialize RoPE rotation matrices
    rope = ll.rope.create(config, n)

    #
    # Thens
    #

    # rotation matrices should have shape (n, d_head)
    assert rope.cos.shape == (n, config.d_head)
    assert rope.sin.shape == (n, config.d_head)

    # rotation matrices should match config dtype
    assert rope.cos.dtype == config.dtype
    assert rope.sin.dtype == config.dtype


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


def test_rotate():
    """Verify RoPE rotation.

    RoPE rotates each pair of values in an embedding by m*theta_i, where m is the embedding's position in the sequence
    and theta_i is based on the ith pair: theta_i = base ** (-2 * i / d).

    To make the math easy, we set i=1, d=4, and base=4/pi**2 which generates theta_1 = pi/2. From here, we can predict
    the rotated values by starting with coordinates (1, 0) and rotating the point in steps of pi/2.
    """

    #
    # Givens
    #

    # I initialized rope matrices for Theta = 4/pi^2, d = 4, and n = 5
    rope_theta = 4 / jnp.pi**2
    d_head = 4
    n = 5

    config = ll.checkpoint.load_config(
        "Llama3.2-3B",
        rope_theta=rope_theta,
        d_head=d_head,
    )
    rope = ll.rope.create(config, n)

    # I generated a sequence of 5 embeddings where the second pair of values (i=1) is (1, 0).
    x = jnp.array(
        [
            [
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ],
        ],
        dtype=jax.dtypes.bfloat16,
    )

    #
    # Whens
    #

    # I rotate x
    x = ll.rope.rotate(rope, x)

    # I drop the batch dimension and round to 2 decimal places
    x = jnp.round(x[0], decimals=2)

    #
    # Thens
    #

    # Embeddings should be rotated in increments of pi/2

    # 0
    assert (x[0, 2], x[0, 3]) == (1, 0)

    # pi/2
    assert (x[1, 2], x[1, 3]) == (0, 1)

    # pi
    assert (x[2, 2], x[2, 3]) == (-1, 0)

    # 3pi/2
    assert (x[3, 2], x[3, 3]) == (0, -1)

    # 2pi
    assert (x[4, 2], x[4, 3]) == (1, 0)
