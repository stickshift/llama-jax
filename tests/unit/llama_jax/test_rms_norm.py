from jax import numpy as jnp
from jax import random

import llama_jax as ll
from llama_jax.rms_norm import RMSNorm


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

    # I create RMSNorm for layers.0.attention_norm
    norm = ll.rms_norm.create(config, params, "layers.0.attention_norm")

    #
    # Thens
    #

    # norm should be populated
    assert norm.weight.shape == (config.d_model,)
    assert norm.eps == config.rms_norm_eps


def test_rms_norm_identity():
    """Verify normalizing an array of ones doesn't change the array."""
    #
    # Givens
    #

    # d_model is 100
    d_model = 100

    # I created RMSNorm with weights of 1.0 and epsilon of 0
    norm = RMSNorm(weight=jnp.ones(d_model), eps=0.0)

    # x is an array of ones
    x = jnp.ones(d_model)

    #
    # Whens
    #

    # I normalize x
    y = ll.rms_norm.forward(norm, x)

    #
    # Thens
    #

    # y should equal x
    assert (y == x).all()


def test_rms_norm_scaling():
    """Verify RMS normalization is invariant to scaling."""
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # d_model is 100
    d_model = 100

    # I created RMSNorm with weights of 1.0 and epsilon of 0
    norm = RMSNorm(weight=jnp.ones(d_model), eps=0.0)

    # x is normally distributed w/ mean of 100 and std of 10
    key, subkey = random.split(key)
    x = 10 * random.normal(subkey, (d_model,)) + 100

    #
    # Whens
    #

    # I normalize x
    y0 = ll.rms_norm.forward(norm, x)

    # I normalize 100*x
    y1 = ll.rms_norm.forward(norm, 100 * x)

    #
    # Thens
    #

    # y1 should approx equal y0
    assert ((y1 - y0) < 1e-2).all()
