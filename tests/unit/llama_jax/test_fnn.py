from jax import Array

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters

from tests.fixtures.jax_fixtures import assert_similar_arrays


def test_factory(config: ModelConfig, params: ModelParameters):
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
    assert ffn.gate.shape == (config.d_model, config.d_ffn)
    assert ffn.output.shape == (config.d_ffn, config.d_model)


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
