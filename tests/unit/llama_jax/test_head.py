from jax import random
import pytest

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

    # I create Head
    head = ll.head.create(config, params)

    #
    # Thens
    #

    # head should be populated
    assert head.output.shape == (config.d_model, config.vocab_size)


@pytest.mark.parametrize("input_shape", [(10, 3072), (2, 10, 3072)])
def test_forward(input_shape: tuple):
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized Head
    params = ll.checkpoint.load_parameters(config)
    head = ll.head.create(config, params)

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, input_shape)

    #
    # Whens
    #

    # I transform embeddings into token logits
    y = ll.head.forward(config, head, x)

    #
    # Thens
    #

    # y.shape should be (config.vocab_size,) for single inputs
    if len(input_shape) == 2:
        assert y.shape == (config.vocab_size,)

    # y.shape should be (bs, config.vocab_size,) for batched inputs
    if len(input_shape) > 2:
        assert y.shape == (input_shape[-3], config.vocab_size)
