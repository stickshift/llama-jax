from jax import Array

from llama_jax.tokenizer import Tokenizer


def test_codec(tokenizer: Tokenizer):
    #
    # Givens
    #

    # A prompt
    prompt0 = "What is the capital of Massachusetts? Answer in one word."

    #
    # Whens
    #

    # I encode prompt
    token_ids = tokenizer.encode(prompt0, bos=False)

    #
    # Thens
    #

    # token_ids should be an Array
    assert isinstance(token_ids, Array)

    #
    # Whens
    #

    # I decode token ids
    prompt1 = tokenizer.decode(token_ids)

    #
    # Thens
    #

    # prompt1 should equal prompt0
    assert prompt1 == prompt0
