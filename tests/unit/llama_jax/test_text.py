from jax import Array

import llama_jax as ll


def test_323b(key: Array):
    #
    # Givens
    #

    # Sequence prompts of mixed length
    prompts = (
        "A B C",
        "one two three four",
    )

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a text generator w/ token sampling disabled and max_tokens = 3
    generator, key = ll.text.generator(config, key, temperature=0, max_tokens=3)

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
        "one two three four five six seven",
    )
