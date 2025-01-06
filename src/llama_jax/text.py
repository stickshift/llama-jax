"""Text completions."""

from collections.abc import Iterator, Sequence
from functools import partial
from typing import Any, Callable

from jax import Array, random

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "generator",
]

TextGenerator = Callable[[str | Sequence[str]], Iterator[str | Sequence[str]]]


def generator(
    config: ModelConfig,
    key: Array,
    *,
    model: Model | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> tuple[TextGenerator, Array]:
    """Create a text generator."""
    # Defaults
    max_tokens = default_arg(max_tokens, 32)

    # Initialize tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Initialize model if not provided
    if model is None:
        model = ll.model.create(config, ll.checkpoint.load_parameters(config))

    # Define generator callable
    def wrapper(prompts: str | Sequence[str], *, key: Array, **kwargs: Any) -> Iterator[str | Sequence[str]]:
        # Override ctor args
        nonlocal max_tokens
        max_tokens = kwargs.get("max_tokens", max_tokens)

        # Remember if prompts are batched
        batched = not isinstance(prompts, str)

        # Initialize key/value cache
        kv_cache = ll.kv_cache.create(config)

        # Split prompts into tokens
        token_ids = tokenizer.encode(prompts)

        # Initialize x with entire sequence on first pass
        x = token_ids

        # Sample up to max tokens
        for _ in range(max_tokens):
            # Transform x into logits
            logits, kv_cache = ll.model.forward(config, model, x, kv_cache=kv_cache)

            # Sample next token
            next_token_id, key = ll.model.next_token(logits, key, temperature=temperature, top_k=top_k, top_p=top_p)

            # Yield next token
            tokens = tokenizer.decode(next_token_id)
            yield tokens if batched else tokens[0]

            # Subsequent iterations process one token at a time
            x = next_token_id

    # Generate subkey to be consumed by wrapper
    key, subkey = random.split(key)
    wrapper = partial(wrapper, key=subkey)

    return wrapper, key
