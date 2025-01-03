import pytest
from jax import Array, random
from jax import numpy as jnp
from jax.nn import softmax

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer
from llama_jax.tokenizer import Tokenizer

from tests.fixtures.jax_fixtures import assert_similar_arrays, similarity_scores


def test_factory(config: ModelConfig, params: ModelParameters):
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


def test_forward(
    config: ModelConfig,
    params: ModelParameters,
    bs: int,
    n: int,
    token_ids: Array,
    logits: Array,
):
    #
    # Givens
    #

    # I initialize x w/ token ids
    x = token_ids

    # I created a Model
    model = ll.model.create(config, params)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    #
    # Whens
    #

    # I transform x into logits
    x, kv_cache = ll.model.forward(config, model, kv_cache, x)

    #
    # Thens
    #

    # kv_cache should be populated
    for i in range(config.n_layers):
        assert kv_cache[i].keys.shape == (bs, config.n_kv_heads, n, config.d_head)
        assert kv_cache[i].values.shape == (bs, config.n_kv_heads, n, config.d_head)

    # x should match expected logits
    assert_similar_arrays(x, logits)


def test_sample_top_k(config: ModelConfig, bs: int, key: Array):
    #
    # Givens
    #

    # I generate random sample probs
    key, subkey = random.split(key)
    probs = softmax(random.normal(subkey, (bs, config.vocab_size)), axis=-1)

    # top_k is 10
    top_k = 10

    #
    # Whens
    #

    # I sample top_k probs
    probs = ll.model.sample_top_k(probs, top_k=top_k)

    #
    # Thens
    #

    # probs.shape should be (bs, top_k)
    assert probs.shape == (bs, top_k)


def test_sample_top_p():
    #
    # Givens
    #

    # I generate 2 batches of sample probs:
    #   0.1, 0.2, 0.3, 0.4
    #   0.05, 0.1, 0.15, 0.7
    probs = jnp.stack([jnp.array([0.1, 0.2, 0.3, 0.4]), jnp.array([0.05, 0.1, 0.15, 0.7])])

    #
    # Whens
    #

    # I sample probs w/ top_p = 0.1
    probs = ll.model.sample_top_p(probs, top_p=0.1)

    #
    # Thens
    #

    # Probs should be
    #   0.1, 0.0, 0.0, 0.0
    #   0.05, 0.1, 0.0, 0.0
    assert jnp.allclose(
        probs,
        jnp.stack([jnp.array([0.1, 0.0, 0.0, 0.0]), jnp.array([0.05, 0.1, 0.0, 0.0])]),
        atol=0.01,
    )


def test_sample_tokens(config: ModelConfig, key: Array):
    #
    # Givens
    #

    # I generated 2 batches of logits
    logits = jnp.stack([
        # First one favors token 0
        jnp.concat([jnp.array([1]), jnp.zeros(config.vocab_size - 1)]),
        # Second one favors token 1
        jnp.concat([jnp.array([0, 1]), jnp.zeros(config.vocab_size - 2)]),
    ])

    #
    # Whens
    #

    # I sample tokens
    key, subkey = random.split(key)
    next_token_ids = ll.model.sample_tokens(logits, key=subkey)

    #
    # Thens
    #

    # next_token_ids.shape should be (2, 1)
    assert next_token_ids.shape == (2, 1)

    # next_token_ids[0] should be 0
    assert next_token_ids[0][0] == 0

    # next_token_ids[1] should be 1
    assert next_token_ids[1][0] == 1


def test_generate(
    config: ModelConfig,
    params: ModelParameters,
    tokenizer: Tokenizer,
    bs: int,
    n: int,
    token_ids: Array,
    key: Array,
):
    #
    # Givens
    #

    # I created a Model
    model = ll.model.create(config, params)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    # Max tokens
    max_tokens = 5

    #
    # Whens
    #

    # Initialize x with entire sequence on first pass
    x = token_ids

    # I sample up to max tokens
    for _ in range(max_tokens):
        # Transform tokens
        logits, kv_cache = ll.model.forward(config, model, kv_cache, x)

        # Sample next token
        key, subkey = random.split(key)
        next_token_id = ll.model.sample_tokens(logits, key=subkey)
        token_ids = jnp.concat([token_ids, next_token_id], axis=-1)

        # Subsequent iterations process one token at a time
        x = next_token_id

        # Break on stop tokens
        if all(v in tokenizer.stop_tokens for v in next_token_id.flatten()):
            break

    #
    # Thens
    #

    # there should be n + max_tokens tokens
    assert token_ids.shape[-1] == n + max_tokens

    # kv_cache should be populated
    for i in range(config.n_layers):
        assert kv_cache[i].keys.shape == (bs, config.n_kv_heads, token_ids.shape[-1] - 1, config.d_head)
        assert kv_cache[i].values.shape == (bs, config.n_kv_heads, token_ids.shape[-1] - 1, config.d_head)
