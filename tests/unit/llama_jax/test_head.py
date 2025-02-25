from jax import Array
import jax.dtypes

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, ModelParameters


def test_factory(params: ModelParameters):
    #
    # Givens
    #

    # I created custom config
    config = ll.checkpoint.load_config(
        "Llama3.2-3B",
        dtype=jax.dtypes.bfloat16,
    )

    #
    # Whens
    #

    # I create Head
    head = ll.head.create(config, params)

    #
    # Thens
    #

    # head should be populated
    assert head.output.shape == (config.d_model, config.vocab_size)
    assert head.output.dtype == config.dtype


def test_forward(config: ModelConfig, params: ModelParameters, bs: int, token_embeddings: Array, position_mask: Array):
    #
    # Givens
    #

    # I initialized Head
    head = ll.head.create(config, params)

    # Sample embeddings
    x = token_embeddings

    #
    # Whens
    #

    # I transform embeddings into token logits
    y = ll.head.forward(config, head, x, position_mask)

    #
    # Thens
    #

    # y.shape should be (bs, config.vocab_size)
    assert y.shape == (bs, config.vocab_size)
