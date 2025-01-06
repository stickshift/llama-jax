from jax import Array, random
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.rms_norm import RMSNorm

from tests.fixtures.jax_fixtures import assert_similar_arrays


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Givens
    #

    # I overrode config dtype
    config = config._replace(dtype=jnp.int32)

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
    assert norm.weight.dtype == config.dtype


def test_rms_norm_shape(config: ModelConfig, bs: int, n: int):
    """Verify normalizing factor is calculated separately for each sample."""

    #
    # Givens
    #

    # I created RMSNorm
    norm = RMSNorm(weight=jnp.ones(config.d_model))

    # x is array of ones
    x = jnp.reshape(jnp.ones(bs * n * config.d_model), (bs, n, config.d_model))

    #
    # Whens
    #

    # I calculate normalizing factor
    factor = ll.rms_norm._norm(config, norm, x)

    #
    # Thens
    #

    # factor should have shape (bs, n, d_model)
    assert factor.shape == (bs, n, config.d_model)


def test_rms_norm_identity(bs: int, n: int):
    """Verify normalizing an array of ones doesn't change the array."""
    #
    # Givens
    #

    # I created RMSNorm with weights of 1.0, epsilon of 0
    config = ll.checkpoint.load_config("Llama3.2-3B", rms_norm_eps=0.0)
    norm = RMSNorm(weight=jnp.ones(config.d_model))

    # x is array of ones
    x = jnp.reshape(jnp.ones(bs * n * config.d_model), (bs, n, config.d_model))

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


def test_rms_norm_scaling(config: ModelConfig, bs: int, n: int, key: Array):
    """Verify RMS normalization is invariant to scaling."""

    #
    # Givens
    #

    # I created RMSNorm with weights of 1.0
    norm = RMSNorm(weight=jnp.ones(config.d_model))

    # x is normally distributed w/ mean of 100 and std of 10
    key, subkey = random.split(key)
    x = 10 * random.normal(subkey, (bs, n, config.d_model)) + 100

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


def test_forward(config: ModelConfig, params: ModelParameters, token_embeddings: Array, attention_norm0: Array):
    #
    # Givens
    #

    # I created RMSNorm from layer 0 attention
    norm = ll.rms_norm.create(config, params, "layers.0.attention_norm")

    # I initialize x from token embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I normalize x
    x = ll.rms_norm.forward(config, norm, x)

    #
    # Thens
    #

    # x should match expected attention_norm0
    assert_similar_arrays(x, attention_norm0)
