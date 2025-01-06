from jax import Array, random

import llama_jax as ll
from llama_jax.tokenizer import Tokenizer


def test_323b(key: Array, tokenizer: Tokenizer):
    #
    # Givens
    #

    # Sequence prompts
    prompts = (
        "A B C",
        "one two three",
    )

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a text generator w/ token sampling disabled and max_tokens = 3
    key, subkey = random.split(key)
    generator = ll.text.generator(config, key=subkey, temperature=0, max_tokens=3)

    #
    # Whens
    #

    # I generate tokens
    for tokens in generator(prompts):
        prompts = tuple(p + t for p, t in zip(prompts, tokens))

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six",
    )
