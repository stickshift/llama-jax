from jax import numpy as jnp
from jax import random
import pytest

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


def test_forward(bs: int, n: int):
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created a Model
    model = ll.model.create(config, params)

    # I generated sample token_ids
    key, subkey = random.split(key)
    token_ids = random.uniform(subkey, (bs, n), maxval=config.vocab_size).astype(jnp.int32)

    #
    # Whens
    #

    # I transform token_ids
    output = ll.model.forward(config, model, token_ids)

    #
    # Thens
    #

    # logits.shape should be (bs, config.vocab_size)
    assert output.logits.shape == (bs, config.vocab_size)


@pytest.mark.wip
def test_generate():
    #
    # Givens
    #

    # rng key
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")
    params = ll.checkpoint.load_parameters(config)

    # I created a Model
    model = ll.model.create(config, params)

    # I split greek prompt into tokens
    prompt = "alpha beta gamma"
    tokenizer = ll.checkpoint.load_tokenizer(config)
    token_ids = tokenizer.encode(prompt)

    #
    # Whens
    #

    # Initialize x with entire sequence on first pass
    x = token_ids

    # I sample 5 tokens
    for _ in range(5):
        # Transform tokens
        output = ll.model.forward(config, model, x)

        # Sample next token
        next_token_id, key = ll.head.sample_token(output.logits, key=key)
        token_ids = jnp.concat([token_ids, next_token_id])

        # Subsequent iterations process one token at a time
        x = next_token_id

    # I decode entire sequence
    content = tokenizer.decode(token_ids)

    #
    # Thens
    #

    # content should be "alpha beta gamma delta epsilon zeta eta"
    assert content.strip() == "alpha beta gamma delta epsilon zeta eta"
