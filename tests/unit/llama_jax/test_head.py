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

    # I create Head
    head = ll.head.create(config, params)

    #
    # Thens
    #

    # head should be populated
    assert head.output.shape == (config.d_model, config.vocab_size)


def test_forward(bs: int, n: int):
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
    x = random.normal(subkey, (bs, n, config.d_model))

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
