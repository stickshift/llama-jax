from jax import random
import pytest

import llama_jax as ll


@pytest.mark.wip
def test_323b():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a text generator w/ token sampling disabled
    key, subkey = random.split(key)
    generator = ll.text.generator(config, key=subkey, temperature=0)

    # Greek prompt
    prompt = "alpha beta gamma"

    #
    # Whens
    #

    # I generate next token
    token = next(generator(prompt))

    #
    # Thens
    #

    # token should be "delta"
    assert token.strip() == "delta"
