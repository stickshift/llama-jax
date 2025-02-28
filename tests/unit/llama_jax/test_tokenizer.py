from llama_jax.tokenizer import Tokenizer


def test_codec(tokenizer: Tokenizer):
    #
    # Givens
    #

    # A prompt
    prompts0 = ("What is the capital of Massachusetts? Answer in one word.",)

    #
    # Whens
    #

    # I encode prompt
    token_ids, _ = tokenizer.encode(prompts0, bos=False)

    #
    # Thens
    #

    # token_ids and position_mask should be a 2D Arrays
    assert token_ids.ndim == 2

    #
    # Whens
    #

    # I decode token ids
    prompts1 = tokenizer.decode(token_ids)

    #
    # Thens
    #

    # prompts1 should equal prompts0
    assert prompts1 == prompts0


def test_padding(tokenizer: Tokenizer):
    #
    # Givens
    #

    # Prompts with different lengths
    prompts = (
        "alpha beta",
        "alpha beta gamma",
    )

    #
    # Whens
    #

    # I encode prompts
    token_ids, position_mask = tokenizer.encode(prompts, bos=False)

    #
    # Thens
    #

    # token_ids should have shape (2, 3)
    assert token_ids.shape == (2, 3)

    # token_ids should be padded
    assert token_ids[0, 2] == tokenizer.pad_id

    # first mask should be [1, 1, 0]
    assert position_mask[0].tolist()[:3] == [1, 1, 0]

    # second mask should be [1, 1, 1]
    assert position_mask[1].tolist()[:3] == [1, 1, 1]


def test_bos(tokenizer: Tokenizer):
    #
    # Givens
    #

    # Prompt
    prompts = ("What is the capital of Massachusetts? Answer in one word.",)

    #
    # Whens
    #

    # I encode prompts with marker
    token_ids, position_mask = tokenizer.encode(prompts, bos=True)

    #
    # Thens
    #

    # token_ids should include bos
    assert token_ids[0, 0] == tokenizer.bos_id

    # position_mask should be all 1s
    assert (position_mask == 1).all()

    #
    # Whens
    #

    # I encode prompts with no marker
    token_ids, position_mask = tokenizer.encode(prompts, bos=False)

    #
    # Thens
    #

    # token_ids should not include bos
    assert token_ids[0, 0] != tokenizer.bos_id

    # position_mask should be all 1s
    assert (position_mask == 1).all()


def test_eos(tokenizer: Tokenizer):
    #
    # Givens
    #

    # Prompt
    prompts = ("What is the capital of Massachusetts? Answer in one word.",)

    #
    # Whens
    #

    # I encode prompts with marker
    token_ids, position_mask = tokenizer.encode(prompts, eos=True)

    #
    # Thens
    #

    # token_ids should include marker
    assert token_ids[0, -1] == tokenizer.eos_id

    # position_mask should be all 1s
    assert (position_mask == 1).all()

    #
    # Whens
    #

    # I encode prompts with no marker
    token_ids, position_mask = tokenizer.encode(prompts, eos=False)

    #
    # Thens
    #

    # token_ids should not include marker
    assert token_ids[0, -1] != tokenizer.eos_id

    # position_mask should be all 1s
    assert (position_mask == 1).all()
