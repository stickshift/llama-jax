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
    config = ll.model.load_config(checkpoint)

    #
    # Thens
    #

    # d_model should be 3072
    assert config.d_model == 3072


def test_load_model():
    #
    # Givens
    #

    # Llama 3.2 3B checkpoint
    checkpoint = "Llama3.2-3B"

    # I loaded model config
    config = ll.model.load_config(checkpoint)

    #
    # Whens
    #

    # I load model
    model = ll.model.load_model(config)

    #
    # Thens
    #

    # embeddings should be populated
    assert model.embeddings.shape == (config.vocab_size, config.d_model)

    # layers should be populated
    assert len(model.layers) == config.n_layers

    # head should be populated
    assert model.head.norm.weight.shape == (config.d_model,)
