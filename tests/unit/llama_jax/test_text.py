from jax import Array, random

import llama_jax as ll
from llama_jax.tokenizer import Tokenizer


def test_323b(key: Array, tokenizer: Tokenizer):
    #
    # Givens
    #

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a text generator w/ token sampling disabled
    key, subkey = random.split(key)
    generator = ll.text.generator(config, key=subkey, temperature=0)

    # Greek prompt
    prompts = ["alpha beta gamma"]

    #
    # Whens
    #

    # I generate next token
    tokens = next(generator(prompts))

    #
    # Thens
    #

    # token should be "delta"
    assert tokens[0].strip() == "delta"
