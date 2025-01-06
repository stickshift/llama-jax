from jax import Array
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters

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

    # I create FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    #
    # Thens
    #

    # ffn should be populated
    assert ffn.input.shape == (config.d_model, config.d_ffn)
    assert ffn.input.dtype == config.dtype
    assert ffn.gate.shape == (config.d_model, config.d_ffn)
    assert ffn.gate.dtype == config.dtype
    assert ffn.output.shape == (config.d_ffn, config.d_model)
    assert ffn.output.dtype == config.dtype


def test_forward(
    config: ModelConfig,
    params: ModelParameters,
    attention_output: Array,
    ffn_output: Array,
):
    #
    # Givens
    #

    # I created FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    # Sample attention output
    x = attention_output

    #
    # Whens
    #

    # I transform x w/ ffn
    x = ll.ffn.forward(config, ffn, x)

    #
    # Thens
    #

    # x should match expected output
    assert_similar_arrays(x, ffn_output)
