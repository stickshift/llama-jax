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


def test_forward():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created Head
    head = ll.head.create(config, params)

    # sequence length
    n = 10

    # I generated sample embeddings
    key, subkey = random.split(key)
    x = random.normal(subkey, (n, config.d_model))

    #
    # Whens
    #

    # I transform embeddings into token logits
    y = ll.head.forward(head, x)

    #
    # Thens
    #

    # y.shape should be vocab_size
    assert y.shape == (config.vocab_size,)
