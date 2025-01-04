from jax import Array, dlpack, random
from jax import numpy as jnp
from jax.nn import softmax
import numpy as np
import torch

import llama_jax as ll
from llama_jax.benchmarks.llama_models import Transformer
from llama_jax.checkpoint import ModelConfig, ModelParameters
from llama_jax.embeddings import Embeddings
from llama_jax.head import Head
from llama_jax.layer import Layer
from llama_jax.tokenizer import Tokenizer

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


def test_forward_w_cache(
    config: ModelConfig, params: ModelParameters, tokenizer: Tokenizer, torch_device, reference_model: Transformer
):
    #
    # Givens
    #

    # I created a Model
    model = ll.model.create(config, params)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    # I initialize token_ids
    token_ids = tokenizer.encode(["one two three four five six seven eight"])

    #
    # Whens
    #

    # I map "<bos> one two three" to logits using reference model
    x = token_ids[:, 0:4]
    logits0 = reference_model(torch.tensor(np.array(x), device=torch_device))[:, -1]
    logits0 = dlpack.from_dlpack(logits0.cpu())

    # I map "<bos> one two three" to logits using empty cache
    x = token_ids[:, 0:4]
    logits1, kv_cache = ll.model.forward(config, model, kv_cache, x)

    #
    # Thens
    #

    # Logits for last embedding vector should match
    assert_similar_arrays(logits0, logits1)

    #
    # Whens
    #

    # I map "<bos> one two three four" to logits using reference model
    x = token_ids[:, 0:5]
    logits0 = reference_model(torch.tensor(np.array(x), device=torch_device))[:, -1]
    logits0 = dlpack.from_dlpack(logits0.cpu())

    # I map "<bos> one two three four" to logits using full cache
    x = token_ids[:, 4:5]
    logits1, kv_cache = ll.model.forward(config, model, kv_cache, x)

    #
    # Thens
    #

    # Logits for last embedding vector should match
    assert_similar_arrays(logits0, logits1)

    #
    # Whens
    #

    # I map "<bos> one two three four five" to logits using reference model
    x = token_ids[:, 0:6]
    logits0 = reference_model(torch.tensor(np.array(x), device=torch_device))[:, -1]
    logits0 = dlpack.from_dlpack(logits0.cpu())

    # I map "<bos> one two three four five" to logits using full cache
    x = token_ids[:, 5:6]
    logits1, kv_cache = ll.model.forward(config, model, kv_cache, x)

    #
    # Thens
    #

    # Logits for last embedding vector should match
    assert_similar_arrays(logits0, logits1)


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
    probs = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.05, 0.1, 0.15, 0.7],
    ])

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
    assert (probs[0] == jnp.array([0.1, 0.0, 0.0, 0.0])).all()
    assert (probs[1] == jnp.array([0.05, 0.1, 0.0, 0.0])).all()


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


def test_generate_wo_sampling(config: ModelConfig, params: ModelParameters, tokenizer: Tokenizer):
    #
    # Givens
    #

    # I created a Model
    model = ll.model.create(config, params)

    # I created a key/value cache
    kv_cache = ll.kv_cache.create(config)

    # Sequence prompts
    prompts = [
        "A B C D",
        "one two three four",
    ]

    # I split prompts into token ids
    token_ids = tokenizer.encode(prompts)

    #
    # Whens
    #

    # I use entire sequence on first pass
    x = token_ids

    # I generate 5 tokens
    for _ in range(5):
        # Transform x into logits
        logits, kv_cache = ll.model.forward(config, model, kv_cache, x)

        # Use most likely next token
        next_token_id = jnp.argmax(logits, axis=-1, keepdims=True)

        # Append next token
        token_ids = jnp.concat([token_ids, next_token_id], axis=-1)

        # Use single token on subsequent passes
        x = next_token_id

    # I decode token_ids
    text = tokenizer.decode(token_ids, strip_special=True)

    #
    # Thens
    #

    # text should be
    assert text == (
        "A B C D E F G H I",
        "one two three four five six seven eight nine",
    )
