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

    # I create Embeddings
    embeddings = ll.embeddings.create(config, params)

    #
    # Thens
    #

    # embeddings should be populated
    assert embeddings.values.shape == (config.vocab_size, config.d_model)


def test_forward():

    #
    # Givens
    #

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I generated sample token_ids
    tokenizer = ll.checkpoint.load_tokenizer(config)
    token_ids = tokenizer.encode("alpha beta gamma")

    # n is length of token_ids
    n = len(token_ids)

    # I created Embeddings
    embeddings = ll.embeddings.create(config, params)

    #
    # Whens
    #

    # I map token ids to embeddings
    x = ll.embeddings.forward(embeddings, token_ids)

    #
    # Thens
    #

    # x is n x d_model array
    assert x.shape == (n, config.d_model)

