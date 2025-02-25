from jax import Array, dlpack
import numpy as np
import torch

from llama_jax.benchmarks import llama_models as llm
from llama_jax.benchmarks.llama_models import Transformer
from llama_jax.checkpoint import ModelConfig
from llama_jax.tokenizer import Tokenizer


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
    logits = transformer(x)

    #
    # Thens
    #

    # logits.shape should be (bs, n, vocab_size)
    assert logits.shape == (bs, n, config.vocab_size)


def test_generate_wo_sampling(config: ModelConfig, torch_device, tokenizer: Tokenizer):
    #
    # Givens
    #

    # Torch device
    device = torch_device

    # I created a llama-models transformer
    transformer = Transformer(config, device)
    transformer.load_state_dict(llm.load_parameters(config, device=device))

    # Sequence prompts
    prompts = [
        "A B C D",
        "one two three four",
    ]

    # I split prompts into token ids
    tids, _ = tokenizer.encode(prompts)
    token_ids = torch.tensor(np.array(tids), device=device)

    #
    # Whens
    #

    # I generate 5 tokens
    for _ in range(5):
        # Transform token_ids into logits
        logits = transformer(token_ids)

        # Use most likely next token
        next_token_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

        # Append next token
        token_ids = torch.concat([token_ids, next_token_id], dim=-1)

    # I decode token_ids
    token_ids = dlpack.from_dlpack(token_ids.cpu())
    text = tokenizer.decode(token_ids, special=False)

    #
    # Thens
    #

    # text should be
    assert text == (
        "A B C D E F G H I",
        "one two three four five six seven eight nine",
    )
