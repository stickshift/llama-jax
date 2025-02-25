import llama_jax as ll


def test_323b():
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

    # I initialized a text generator w/ token sampling disabled
    generator = ll.text.generator(config, temperature=0)

    #
    # Whens
    #

    # I generate 3 tokens
    for tokens in generator(prompts, max_tokens=3):
        prompts = tuple(p + t for p, t in zip(prompts, tokens))

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six seven",
    )


def test_318b():
    #
    # Givens
    #

    # Sequence prompts of mixed length
    prompts = (
        "A B C",
        "one two three four",
    )

    # I loaded config for 3.1 8B checkpoint
    config = ll.checkpoint.load_config("Llama3.1-8B")

    # I initialized a text generator w/ token sampling disabled
    generator = ll.text.generator(config, temperature=0)

    #
    # Whens
    #

    # I generate 3 tokens
    for tokens in generator(prompts, max_tokens=3):
        prompts = tuple(p + t for p, t in zip(prompts, tokens))

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six seven",
    )
