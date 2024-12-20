import llama_jax as ll
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer


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

    # I create Model
    model = ll.model.create(config, params)

    #
    # Thens
    #

    # embeddings should be populated
    assert isinstance(model.embeddings, Embeddings)

    # layers should be populated
    assert len(model.layers) == config.n_layers
    for layer in model.layers:
        assert isinstance(layer, Layer)

    # head should be populated
    assert isinstance(model.head, Head)


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

    # I created a Model
    model = ll.model.create(config, params)

    #
    # Whens
    #

    # I transform token_ids into next token logits
    y = ll.model.forward(model, token_ids)

    #
    # Thens
    #

    # y.shape should be vocab_size
    assert y.shape == (config.vocab_size,)
