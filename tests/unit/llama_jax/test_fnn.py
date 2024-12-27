from jax import Array

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters


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


def test_forward(config: ModelConfig, params: ModelParameters, token_embeddings: Array):
    #
    # Givens
    #

    # I created FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform x w/ ffn
    y = ll.ffn.forward(config, ffn, x)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
