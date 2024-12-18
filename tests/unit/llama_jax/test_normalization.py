from jax import numpy as jnp

import llama_jax as ll
from llama_jax.normalization import RMSNorm


def test_rms_norm_identity():
    """Verify normalizing an array of ones doesn't change the array."""
    #
    # Givens
    #

    # d_model is 100
    d_model = 100

    # I created RMSNorm with weights of 1.0 and epsilon of 0
    norm = RMSNorm(weight=jnp.ones(d_model), eps=0.0)

    # x is an array of 100 ones
    x = jnp.ones(100)

    #
    # Whens
    #

    # I normalize x
    y = ll.normalization.rms(norm, x)

    #
    # Thens
    #

    # y should be equal to x
    assert (y == x).all()
