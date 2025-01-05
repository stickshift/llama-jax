from collections.abc import Sequence

from jax import Array
from jax import numpy as jnp
from pytest import approx

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer
from llama_jax.tokenizer import Tokenizer
from llama_jax.tools import default_arg

from tests.fixtures.jax_fixtures import assert_similar_arrays


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


def test_forward_full_sequence(
    config: ModelConfig,
    params: ModelParameters,
    token_ids: Array,
    logits: Array,
):
    """Transforms tokens into logits in a single pass."""

    #
    # Givens
    #

    # Logits for last token in sequence
    logits0 = logits[:, -1]

    # I created a Model
    model = ll.model.create(config, params)

    #
    # Whens
    #

    # I transform entire sequence into logits
    logits1 = ll.model.forward(config, model, token_ids)

    #
    # Thens
    #

    # logits1 should match expected logits0
    assert_similar_arrays(logits1, logits0)


def test_forward_incremental(
    config: ModelConfig,
    params: ModelParameters,
    n: int,
    token_ids: Array,
    logits: Array,
):
    """Transforms tokens into logits one token at a time."""

    #
    # Givens
    #

    # I created a Model
    model = ll.model.create(config, params)

    # I initialized key/value cache
    kv_cache = ll.kv_cache.create(config)

    #
    # Whens
    #

    # I iterate through tokens one at a time
    for i in range(n):
        # I look up current token
        x = token_ids[:, i : i + 1]

        # I look up expected logits for current token
        logits0 = logits[:, i]

        # I transform x into logits
        logits1, kv_cache = ll.model.forward(config, model, x, kv_cache=kv_cache)

        #
        # Thens
        #

        # logits1 should match expected logits0
        assert_similar_arrays(logits1, logits0)


def test_sample_top_k():
    #
    # Givens
    #

    # Sorted array with 2 batches of sample probabilities
    probs = jnp.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.8, 0.1, 0.05, 0.05],
    ])

    #
    # Whens
    #

    # I sample probs w/ top_k = 1
    x = ll.model._sample_top_k(probs, top_k=1)

    #
    # Thens
    #

    # x should be
    #   0.4
    #   0.8
    assert (x[0] == jnp.array([0.4])).all()
    assert (x[1] == jnp.array([0.8])).all()

    #
    # Whens
    #

    # I sample probs w/ top_k = 3
    x = ll.model._sample_top_k(probs, top_k=3)

    #
    # Thens
    #

    # x should be
    #   0.4, 0.3, 0.2
    #   0.8, 0.1, 0.05
    assert (x[0] == jnp.array([0.4, 0.3, 0.2])).all()
    assert (x[1] == jnp.array([0.8, 0.1, 0.05])).all()

    #
    # Whens
    #

    # I sample probs w/ top_k = 10
    x = ll.model._sample_top_k(probs, top_k=10)

    #
    # Thens
    #

    # x should equal probs
    assert (x == probs).all()


def test_sample_top_p():
    #
    # Givens
    #

    # Sorted array with 2 batches of sample probabilities
    probs = jnp.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.8, 0.1, 0.05, 0.05],
    ])

    #
    # Whens
    #

    # I sample probs w/ top_p = 0.1
    x = ll.model._sample_top_p(probs, top_p=0.1)

    #
    # Thens
    #

    # x should be
    #   0.4, 0.0, 0.0, 0.0
    #   0.8, 0.0, 0.0, 0.0
    assert (x[0] == jnp.array([0.4, 0.0, 0.0, 0.0])).all()
    assert (x[1] == jnp.array([0.8, 0.0, 0.0, 0.0])).all()

    #
    # Whens
    #

    # I sample probs w/ top_p = 0.5
    x = ll.model._sample_top_p(probs, top_p=0.5)

    #
    # Thens
    #

    # x should be
    #   0.4, 0.3, 0.0, 0.0
    #   0.8, 0.0, 0.0, 0.0
    assert (x[0] == jnp.array([0.4, 0.3, 0.0, 0.0])).all()
    assert (x[1] == jnp.array([0.8, 0.0, 0.0, 0.0])).all()


def test_next_token_max_logit(key: Array, logits: Array):
    #
    # Whens
    #

    # I select logits for last embedding
    logits = logits[:, -1]

    # I select next token w/ sampling disabled
    next_tokens, key = ll.model.next_token(logits, key, temperature=0)

    #
    # Thens
    #

    # next_tokens.shape should be (2, 1)
    assert next_tokens.shape == (2, 1)

    # next_tokens should be based on max logits
    assert (next_tokens == jnp.argmax(logits, axis=-1, keepdims=True)).all()


def test_next_token_random_sample(key: Array):
    #
    # Givens
    #

    # Number of iterations
    n = 2000

    # Sample routine that counts the number of times each token id is selected
    def sample(probs: Sequence[float], key: Array, temperature: float | None = None) -> tuple[dict, Array]:
        temperature = default_arg(temperature, 1.0)

        # Initialize counts to 0
        counts = dict.fromkeys(range(len(probs)), 0)

        # Convert probs into logits using inverse softmax
        log_probs = jnp.log(jnp.array(probs))
        logits = jnp.array([log_probs - jnp.mean(log_probs)])

        # Sample tokens n times
        for _ in range(n):
            next_token, key = ll.model.next_token(logits, key, temperature=temperature)
            counts[next_token.item()] = counts[next_token.item()] + 1

        return counts, key

    #
    # Whens
    #

    # I sample tokens with uniform probs
    counts, key = sample([1.0, 1.0, 1.0], key)

    #
    # Thens
    #

    # Counts should match probs
    assert counts[0] == approx(n / 3, rel=0.1)
    assert counts[1] == approx(n / 3, rel=0.1)
    assert counts[2] == approx(n / 3, rel=0.1)

    #
    # Whens
    #

    # I sample tokens with ascending probs 1/6, 2/6, 3/6
    counts, key = sample([1.0, 2.0, 3.0], key)

    #
    # Thens
    #

    # Counts should match probs
    assert counts[0] == approx(n / 6, rel=0.1)
    assert counts[1] == approx(n / 3, rel=0.1)
    assert counts[2] == approx(n / 2, rel=0.1)

    #
    # Whens
    #

    # I sample tokens with ascending probs and temperature > 1 (increased randomness)
    counts, key = sample([1.0, 2.0, 3.0], key, temperature=2.0)

    #
    # Thens
    #

    # counts[2] should be less than n / 2
    assert counts[2] < n / 2

    #
    # Whens
    #

    # I sample tokens with ascending probs and temperature < 1 (decreased randomness)
    counts, key = sample([1.0, 2.0, 3.0], key, temperature=0.1)

    #
    # Thens
    #

    # counts[2] should be greater than n / 2
    assert counts[2] > n / 2


def test_generate_wo_cache(config: ModelConfig, params: ModelParameters, tokenizer: Tokenizer, key: Array):
    #
    # Givens
    #

    # Sequence prompts
    prompts = (
        "A B C",
        "one two three",
    )

    # I split prompts into tokens
    tokens = tokenizer.encode(prompts)

    # I created a Model
    model = ll.model.create(config, params)

    #
    # Whens
    #

    # I generate 3 tokens w/ sampling disabled
    for _ in range(3):
        # Transform tokens into logits
        logits = ll.model.forward(config, model, tokens)

        # Sample next token
        next_token, key = ll.model.next_token(logits, key, temperature=0)

        # Process all tokens on next pass
        tokens = jnp.concat([tokens, next_token], axis=-1)

    # I decode tokens
    prompts = tokenizer.decode(tokens, special=False)

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six",
    )


def test_generate_w_cache(config: ModelConfig, params: ModelParameters, tokenizer: Tokenizer, key: Array):
    #
    # Givens
    #

    # Sequence prompts
    prompts = (
        "A B C",
        "one two three",
    )

    # I split prompts into tokens
    tokens = tokenizer.encode(prompts)

    # I initialized key/value cache
    kv_cache = ll.kv_cache.create(config)

    # I created a Model
    model = ll.model.create(config, params)

    #
    # Whens
    #

    # I generate 3 tokens w/ sampling disabled
    x = tokens
    for _ in range(3):
        # Transform x into logits
        logits, kv_cache = ll.model.forward(config, model, x, kv_cache=kv_cache)

        # Sample next token
        next_token, key = ll.model.next_token(logits, key, temperature=0)

        # Update full list of tokens
        tokens = jnp.concat([tokens, next_token], axis=-1)

        # Process generated token on next pass
        x = next_token

    # I decode tokens
    prompts = tokenizer.decode(tokens, special=False)

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six",
    )
