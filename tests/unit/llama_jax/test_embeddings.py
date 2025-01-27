from jax import Array
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters


def test_factory(config: ModelConfig, params: ModelParameters):
    #
    # Givens
    #

    # I overrode config dtype
    config = config._replace(dtype=jnp.int32)

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
    assert embeddings.values.dtype == config.dtype


def test_forward(config: ModelConfig, params: ModelParameters, token_ids: Array, token_embeddings: Array):
    #
    # Givens
    #

    # I created Embeddings
    embeddings = ll.embeddings.create(config, params)

    #
    # Whens
    #

    # I map token ids to embeddings
    x = ll.embeddings.forward(config, embeddings, token_ids)

    #
    # Thens
    #

    # x shape matches expected embeddings
    assert x.shape == token_embeddings.shape

    # x dtype matches expected embeddings
    assert x.dtype == token_embeddings.dtype

    # x values match expected embeddings
    assert (x == token_embeddings).all()
