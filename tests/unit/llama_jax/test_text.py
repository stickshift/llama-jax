from jax import random

import llama_jax as ll


def test_323b():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created a Model
    model = ll.model.create(config, params)

    # Greek prompt
    prompt = "alpha beta gamma"

    # I initialized a text generator w/ token sampling disabled
    key, subkey = random.split(key)
    generator = ll.text.generator(model, key=subkey, temperature=0)

    #
    # Whens
    #

    # I generate next token w/ sampling disabled
    token = next(generator(prompt))

    #
    # Thens
    #

    # token should be "delta"
    assert token.strip() == "delta"
