from jax import numpy as jnp

import llama_jax as ll


def test_load_config():
    #
    # Givens
    #

    # Llama 3.2 3B checkpoint
    checkpoint = "Llama3.2-3B"

    #
    # Whens
    #

    # I load model config
    config = ll.checkpoint.load_config(checkpoint)

    #
    # Thens
    #

    # config should be populated
    assert config.max_tokens == 512
    assert config.vocab_size == 128256
    assert config.d_model == 3072

    # rope rotation matrices should be populated (max_tokens, d_head)
    assert len(config.rope_cos) == config.max_tokens
    assert all(len(v) == config.d_head for v in config.rope_cos)
    assert len(config.rope_sin) == config.max_tokens
    assert all(len(v) == config.d_head for v in config.rope_sin)

    # mask should be populated (max_tokens, max_tokens) w/ with zeros below the diagonal, -inf above the diagonal
    for i in range(config.max_tokens):
        for j in range(config.max_tokens):
            if j > i:
                assert config.mask[i][j] == -jnp.inf
            else:
                assert config.mask[i][j] == 0

    # config should be hashable
    hash(config)


def test_load_parameters():
    #
    # Givens
    #

    # Llama 3.2 3B checkpoint
    checkpoint = "Llama3.2-3B"

    # I loaded model config
    config = ll.checkpoint.load_config(checkpoint)

    #
    # Whens
    #

    # I load model parameters
    params = ll.checkpoint.load_parameters(config)

    #
    # Thens
    #

    # there should be 255 parameters
    assert len(params) == 255
