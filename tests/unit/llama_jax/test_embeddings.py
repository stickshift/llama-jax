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

    # I create Embeddings
    embeddings = ll.embeddings.create(config, params)

    #
    # Thens
    #

    # embeddings should be populated
    assert embeddings.values.shape == (config.vocab_size, config.d_model)


def test_forward():
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created Embeddings
    embeddings = ll.embeddings.create(config, params)

    # bs, n
    bs = 2
    n = 10

    # I generated sample token_ids
    key, subkey = random.split(key)
    token_ids = random.uniform(subkey, (bs, n), maxval=config.vocab_size).astype(jnp.int32)

    #
    # Whens
    #

    # I map token ids to embeddings
    x = ll.embeddings.forward(config, embeddings, token_ids)

    #
    # Thens
    #

    # x is bs x n x d_model array
    assert x.shape == (bs, n, config.d_model)
