import jax
from jax import numpy as jnp
from jax import random

import llama_jax as ll


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

    # I create FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    #
    # Thens
    #

    # ffn should be populated
    assert ffn.input.shape == (config.d_model, config.d_ffn)
    assert ffn.gate.shape == (config.d_model, config.d_ffn)
    assert ffn.output.shape == (config.d_ffn, config.d_model)


def test_forward():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created FFN for layers.0.feed_forward
    ffn = ll.ffn.create(config, params, "layers.0.feed_forward")

    # sequence length
    n = 10

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (n, config.d_model))

    #
    # Whens
    #

    # I transform x w/ ffn
    y = ll.ffn.forward(ffn, x)

    #
    # Thens
    #

    # y.shape didn't change
    assert y.shape == x.shape
