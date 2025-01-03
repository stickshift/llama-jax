import numpy as np
from jax import Array

import torch

from llama_jax.checkpoint import ModelConfig

from llama_jax.benchmarks import llama_models as llm
from llama_jax.benchmarks.llama_models import Transformer


def test_transformer(config: ModelConfig, torch_device, bs: int, n: int, token_ids: Array):
    #
    # Givens
    #

    # Torch device
    device = torch_device

    # I initialize x w/ token ids
    x = torch.tensor(np.array(token_ids), device=device)

    # I create a llama-models transformer
    transformer = Transformer(config, device)
    transformer.load_state_dict(llm.load_parameters(config, device=device))

    #
    # Whens
    #

    # I transform x into logits
    y = transformer(x)

    #
    # Thens
    #

    # y.shape should be (bs, n, vocab_size)
    assert y.shape == (bs, n, config.vocab_size)

