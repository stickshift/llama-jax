import jax.numpy as jnp


def test_array():
    #
    # Givens
    #

    # n = 10
    n = 10

    #
    # Whens
    #

    # I create array of length n
    x = jnp.arange(n)

    #
    # Thens
    #

    # x should have length n
    assert len(x) == n
