from jax import numpy as jnp
from jax import random
import pytest

import llama_jax as ll
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer


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

    # I create Model
    model = ll.model.create(config, params)

    #
    # Thens
    #

    # embeddings should be populated
    assert isinstance(model.embeddings, Embeddings)

    # layers should be populated
    assert len(model.layers) == config.n_layers
    for layer in model.layers:
        assert isinstance(layer, Layer)

    # head should be populated
    assert isinstance(model.head, Head)


@pytest.mark.parametrize(
    "input_shape",
    [
        (10,),
        (
            2,
            10,
        ),
    ],
)
def test_forward(input_shape: tuple):
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created a Model
    model = ll.model.create(config, params)

    # I generated sample token_ids
    key, subkey = random.split(key)
    token_ids = random.uniform(subkey, input_shape, maxval=config.vocab_size).astype(jnp.int32)

    #
    # Whens
    #

    # I transform token_ids into next token logits
    y = ll.model.forward(config, model, token_ids)

    #
    # Thens
    #

    # y.shape should be (config.vocab_size,) for single inputs
    if len(input_shape) == 1:
        assert y.shape == (config.vocab_size,)

    # y.shape should be (bs, config.vocab_size,) for batched inputs
    if len(input_shape) > 1:
        assert y.shape == (input_shape[-2], config.vocab_size)
