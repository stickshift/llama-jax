from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig


def test_position_mask(config: ModelConfig):
    #
    # Givens
    #

    # Seed
    #   1 1 0
    #   1 1 1
    seed = jnp.array([[1, 1, 0], [1, 1, 1]])

    #
    # Whens
    #

    # I create a position mask
    position_mask = ll.position_mask.create(seed, max_tokens=config.max_tokens)

    #
    # Thens
    #

    # dtype should be int32
    assert position_mask.dtype == jnp.int32

    # shape should be (bs, max_tokens)
    assert position_mask.shape == (seed.shape[0], config.max_tokens)

    # First n tokens should match seed
    assert (position_mask[:, : seed.shape[-1]] == seed).all()
