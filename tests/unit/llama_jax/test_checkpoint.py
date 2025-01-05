import llama_jax as ll


def test_load_config():
    #
    # Givens
    #

    # Llama 3.2 3B checkpoint
    checkpoint = "Llama3.2-3B"

    #
    # Whens
    #

    # I load model config
    config = ll.checkpoint.load_config(checkpoint)

    #
    # Thens
    #

    # config should be populated
    assert config.max_tokens == 512
    assert config.vocab_size == 128256
    assert config.d_model == 3072

    # config should be hashable
    hash(config)


def test_load_parameters():
    #
    # Givens
    #

    # Llama 3.2 3B checkpoint
    checkpoint = "Llama3.2-3B"

    # I loaded model config
    config = ll.checkpoint.load_config(checkpoint)

    #
    # Whens
    #

    # I load model parameters
    params = ll.checkpoint.load_parameters(config)

    #
    # Thens
    #

    # there should be 255 parameters
    assert len(params) == 255
