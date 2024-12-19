from jax import random

import llama_jax as ll


def test_323b_text():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created Model
    model = ll.model.create(config, params)

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I generate next token
    key, subkey = random.split(key)
    token = next(ll.text.generate(subkey, config, model, prompt))

    #
    # Thens
    #

    # token should be "delta"
    assert token.strip() == "delta"
