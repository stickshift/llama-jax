from jax import random

import llama_jax as ll
from llama_jax.attention import Attention
from llama_jax.ffn import FFN


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
    assert model.embeddings.shape == (config.vocab_size, config.d_model)

    # layers should be populated
    assert len(model.layers) == config.n_layers

    # head should be populated
    assert model.head.norm.weight.shape == (config.d_model,)
