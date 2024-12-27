"""Text completions."""

from collections.abc import Iterator, Sequence
from functools import partial
from typing import Callable

from jax import random
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig
from llama_jax.model import Model
from llama_jax.tokenizer import Tokenizer
from llama_jax.tools import default_arg

__all__ = [
    "generator",
]


def generator(
    config: ModelConfig,
    *,
    key: ArrayLike,
    model: Model | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> Callable[[Sequence[str]], Iterator[Sequence[str]]]:
    """Create a text generator."""
    # Defaults
    max_tokens = default_arg(max_tokens, 32)

    # Initialize tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Initialize model if not provided
    if model is None:
        model = ll.model.create(config, ll.checkpoint.load_parameters(config))

    return partial(
        _generate,
        config=config,
        key=key,
        tokenizer=tokenizer,
        model=model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _generate(
    prompts: Sequence[str],
    *,
    config: ModelConfig,
    tokenizer: Tokenizer,
    key: ArrayLike,
    model: Model,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_tokens: int,
) -> Iterator[Sequence[str]]:
    """Generate tokens given a prompt."""
    # Initialize key/value cache
    kv_cache = ll.kv_cache.create(config)

    # Split prompts into tokens
    token_ids = tokenizer.encode(prompts)

    # Initialize x with entire sequence on first pass
    x = token_ids

    # I sample up to max tokens
    for _ in range(max_tokens):
        # Transform tokens
        logits, kv_cache = ll.model.forward(config, model, kv_cache, x)

        # Sample next token
        key, subkey = random.split(key)
        next_token_id = ll.model.sample_tokens(logits, key=subkey, temperature=temperature, top_k=top_k, top_p=top_p)

        # Break on stop tokens
        if all(v in tokenizer.stop_tokens for v in next_token_id.flatten()):
            break

        # Yield next token
        yield tokenizer.decode(next_token_id)

        # Subsequent iterations process one token at a time
        x = next_token_id
