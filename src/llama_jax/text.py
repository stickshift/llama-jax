"""Text completions."""

from collections.abc import Iterator, Sequence
from typing import Any, Callable

from jax import Array, random
from jax import numpy as jnp

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, TrainingLevel
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "generator",
]

TextGenerator = Callable[[str | Sequence[str]], Iterator[str | Sequence[str]]]


def generator(
    config: ModelConfig,
    *,
    key: Array | None = None,
    model: Model | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> TextGenerator:
    """Create a text generator."""
    # Defaults
    max_tokens = default_arg(max_tokens, 32)

    # Validate
    if config.training is not TrainingLevel.PRETRAINED:
        raise ValueError("Chat generator requires PRETRAINED model.")

    # Initialize tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Initialize model if not provided
    if model is None:
        model = ll.model.create(config, ll.checkpoint.load_parameters(config))

    # Define generator callable
    def wrapper(prompts: str | Sequence[str], **kwargs: Any) -> Iterator[str | Sequence[str]]:
        nonlocal key, max_tokens, temperature, top_k, top_p

        # Override ctor args
        max_tokens = kwargs.get("max_tokens", max_tokens)
        temperature = kwargs.get("temperature", temperature)
        top_k = kwargs.get("top_k", top_k)
        top_p = kwargs.get("top_p", top_p)

        assert max_tokens is not None

        # Remember if prompts are batched
        batched = not isinstance(prompts, str)

        # Initialize key/value cache
        kv_cache = ll.kv_cache.create(config)

        # Split prompts into tokens
        token_ids, position_mask = tokenizer.encode(prompts)

        # Initialize x with entire sequence on first pass
        x = token_ids

        # All sequences in batch start off active
        active = jnp.ones(x.shape[0], dtype=bool)

        # Sample up to max tokens
        for _ in range(max_tokens):
            # Transform x into logits
            logits, kv_cache = ll.model.forward(config, model, x, position_mask, kv_cache=kv_cache)

            # Sample next token
            key, subkey = random.split(key) if key is not None else (None, None)
            next_token_id = ll.model.next_token(logits, key=subkey, temperature=temperature, top_k=top_k, top_p=top_p)

            # Track active sequences
            is_stop_token = jnp.isin(next_token_id.squeeze(), tokenizer.stop_tokens)
            active = active & ~is_stop_token
            if not jnp.any(active):
                break

            # Yield next token
            tokens = tokenizer.decode(next_token_id, special=False)
            yield tokens if batched else tokens[0]

            # Subsequent iterations process one token at a time.
            #   We multiply by active to explicitly zero out inactive sequences moving forward.
            x = next_token_id * active[:, None]
            position_mask = ll.model.increment_position_mask(position_mask) * active[:, None]

    return wrapper
