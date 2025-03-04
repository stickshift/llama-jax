from collections.abc import Sequence

from jax import Array, random
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
    model = ll.model.create(config, params=params)

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
    position_mask: Array,
    logits: Array,
):
    """Transforms tokens into logits in a single pass."""

    #
    # Givens
    #

    # Logits for last token in sequence
    logits0 = logits[:, -1]

    # I initialize model
    model = ll.model.create(config, params=params)

    #
    # Whens
    #

    # I transform entire sequence into logits
    logits1 = ll.model.forward(config, model, token_ids, position_mask)

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
    position_mask: Array,
    logits: Array,
):
    """Transforms tokens into logits one token at a time."""

    #
    # Givens
    #

    # I created a Model
    model = ll.model.create(config, params=params)

    # I initialized key/value cache
    kvc = ll.kvc.create(config)

    #
    # Whens
    #

    # I iterate through tokens one at a time
    for i in range(n):
        # I look up current token and mask
        x = token_ids[:, i : i + 1]

        # I look up expected logits for current token
        logits0 = logits[:, i]

        # I transform x into logits
        logits1, kvc = ll.model.forward(config, model, x, position_mask, kvc=kvc)

        #
        # Thens
        #

        # logits1 should match expected logits0
        assert_similar_arrays(logits1, logits0)


def test_next_token_max_logit(logits: Array):
    #
    # Whens
    #

    # I select logits for last embedding
    logits = logits[:, -1]

    # I select next token w/ sampling disabled
    next_tokens = ll.model.next_token_id(logits, temperature=0)

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
            key, subkey = random.split(key)
            next_token = ll.model.next_token_id(logits, key=subkey, temperature=temperature)
            counts[next_token.item()] = counts[next_token.item()] + 1

        return counts

    #
    # Whens
    #

    # I sample tokens with uniform probs
    key, subkey = random.split(key)
    counts = sample([1 / 3, 1 / 3, 1 / 3], subkey)

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
    key, subkey = random.split(key)
    counts = sample([1 / 6, 2 / 6, 3 / 6], subkey)

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
    key, subkey = random.split(key)
    counts = sample([1.0, 2.0, 3.0], subkey, temperature=2.0)

    #
    # Thens
    #

    # counts[2] should be less than n / 2
    assert counts[2] < n / 2

    #
    # Whens
    #

    # I sample tokens with ascending probs and temperature < 1 (decreased randomness)
    key, subkey = random.split(key)
    counts = sample([1.0, 2.0, 3.0], subkey, temperature=0.1)

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
    token_ids, position_mask = tokenizer.encode(prompts)

    # I created a Model
    model = ll.model.create(config, params=params)

    #
    # Whens
    #

    # I generate 3 tokens w/ sampling disabled
    for _ in range(3):
        # Transform tokens into logits
        logits = ll.model.forward(config, model, token_ids, position_mask)

        # Sample next token
        key, subkey = random.split(key)
        next_token_id = ll.model.next_token_id(logits, key=subkey, temperature=0)

        # Process all tokens on next pass
        token_ids = jnp.concat([token_ids, next_token_id], axis=-1)

    # I decode tokens
    prompts = tokenizer.decode(token_ids, special=False)

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
    token_ids, position_mask = tokenizer.encode(prompts)

    # I initialized key/value cache
    kvc = ll.kvc.create(config)

    # I created a Model
    model = ll.model.create(config, params=params)

    #
    # Whens
    #

    # I generate 3 tokens w/ sampling disabled
    x = token_ids
    for _ in range(3):
        # Transform x into logits
        logits, kvc = ll.model.forward(config, model, x, position_mask, kvc=kvc)

        # Sample next token
        key, subkey = random.split(key)
        next_token_id = ll.model.next_token_id(logits, key=subkey, temperature=0)

        # Update full list of tokens
        token_ids = jnp.concat([token_ids, next_token_id], axis=-1)

        # Process generated token on next pass
        x = next_token_id

    # I decode tokens
    prompts = tokenizer.decode(token_ids, special=False)

    #
    # Thens
    #

    # prompts should be
    assert prompts == (
        "A B C D E F",
        "one two three four five six",
    )
