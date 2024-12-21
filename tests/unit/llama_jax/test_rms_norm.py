from math import prod

from jax import numpy as jnp
from jax import random
import pytest

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


@pytest.mark.parametrize("input_shape", [(10, 100), (2, 10, 100)])
def test_rms_norm_shape(input_shape: tuple):
    """Verify normalizing factor is calculated separately for each sample.

    Args:
        input_shape: (n, d_model) or (bs, n, d_model)
    """
    #
    # Givens
    #

    # I created RMSNorm with weights of 1.0, epsilon of 0
    config = ll.checkpoint.load_config("Llama3.2-3B", rms_norm_eps=0.0)
    norm = RMSNorm(weight=jnp.ones(input_shape[-1]))

    # x is array of ones
    x = jnp.reshape(jnp.ones(prod(input_shape)), input_shape)

    #
    # Whens
    #

    # I calculate normalizing factor
    factor = ll.rms_norm._norm(config, norm, x)

    #
    # Thens
    #

    # factor should have shape (..., 1)
    assert factor.shape == input_shape[:-1] + (1,)


@pytest.mark.parametrize("input_shape", [(10, 100), (2, 10, 100)])
def test_rms_norm_identity(input_shape: tuple):
    """Verify normalizing an array of ones doesn't change the array.

    Args:
        input_shape: (n, d_model) or (bs, n, d_model)
    """
    #
    # Givens
    #

    # I created RMSNorm with weights of 1.0, epsilon of 0
    config = ll.checkpoint.load_config("Llama3.2-3B", rms_norm_eps=0.0)
    norm = RMSNorm(weight=jnp.ones(input_shape[-1]))

    # x is array of ones
    x = jnp.reshape(jnp.ones(prod(input_shape)), input_shape)

    #
    # Whens
    #

    # I normalize x
    y = ll.rms_norm.forward(config, norm, x)

    #
    # Thens
    #

    # y should equal x
    assert (y == x).all()


@pytest.mark.parametrize("input_shape", [(10, 100), (2, 10, 100)])
def test_rms_norm_scaling(input_shape: tuple):
    """Verify RMS normalization is invariant to scaling.

    Args:
        input_shape: (n, d_model) or (bs, n, d_model)
    """
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # I created RMSNorm with weights of 1.0, epsilon of 0
    config = ll.checkpoint.load_config("Llama3.2-3B", rms_norm_eps=0.0)
    norm = RMSNorm(weight=jnp.ones(input_shape[-1]))

    # x is normally distributed w/ mean of 100 and std of 10
    key, subkey = random.split(key)
    x = 10 * random.normal(subkey, input_shape) + 100

    #
    # Whens
    #

    # I normalize x
    y0 = ll.rms_norm.forward(config, norm, x)

    # I normalize 100*x
    y1 = ll.rms_norm.forward(config, norm, 100 * x)

    #
    # Thens
    #

    # y1 should approx equal y0
    assert ((y1 - y0) < 1e-2).all()
