from jax import Array

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Whens
    #

    # I create Head
    head = ll.head.create(config, params)

    #
    # Thens
    #

    # head should be populated
    assert head.output.shape == (config.d_model, config.vocab_size)


def test_forward(config: ModelConfig, params: ModelParameters, bs: int, token_embeddings: Array):
    #
    # Givens
    #

    # I initialized Head
    head = ll.head.create(config, params)

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform embeddings into token logits
    y = ll.head.forward(config, head, x)

    #
    # Thens
    #

    # y.shape should be (bs, config.vocab_size)
    assert y.shape == (bs, config.vocab_size)
