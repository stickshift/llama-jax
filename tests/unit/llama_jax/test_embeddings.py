import llama_jax as ll
from llama_jax.embeddings import Embeddings


def test_embeddings():

    #
    # Givens
    #

    # I loaded config for 3.2 3B checkpoint
    config = ll.model.load_config("Llama3.2-3B")

    # I split prompt into token_ids
    tokenizer = ll.model.load_tokenizer(config)
    token_ids = tokenizer.encode("alpha beta gamma")

    # n is length of token_ids
    n = len(token_ids)

    #
    # Whens
    #

    # I initialize embeddings
    params = ll.model.load_parameters(config)
    embeddings = Embeddings(values=params[f"tok_embeddings.weight"])

    #
    # Thens
    #

    # embeddings should be populated
    assert embeddings.values.shape == (config.vocab_size, config.d_model)

    #
    # Whens
    #

    # I map token ids to embeddings
    x = ll.embeddings(embeddings, token_ids)

    #
    # Thens
    #

    # x is n x d_model array
    assert x.shape == (n, config.d_model)

