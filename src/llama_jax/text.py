"""Text completions."""

from collections.abc import Iterator

from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.model import Model
from llama_jax.tokenizer import Tokenizer
from llama_jax.tools import default_arg

__all__ = [
    "generate",
]


def generate(
    key: ArrayLike,
    tokenizer: Tokenizer,
    model: Model,
    prompt: str,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Generate tokens given a prompt."""
    # Defaults
    max_tokens = default_arg(max_tokens, 32)

    # Split prompt into tokens
    token_ids = tokenizer.encode(prompt)

    # Convert token ids into mutable list
    token_ids = token_ids.tolist()

    # Generate output until we get a stop token or we exceed max_tokens.
    for _ in range(max_tokens):
        # Initialize x with current token ids
        x = jnp.array(token_ids)

        # Transform token ids into next token logits
        logits = ll.model.forward(model, x)

        # Sample tokens
        key, subkey = random.split(key)
        token_id = ll.head.sample_token(
            subkey,
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Check stopping criteria
        if token_id in tokenizer.stop_tokens:
            break

        # Yield token
        yield tokenizer.decode(token_id)

        # Append to end of sequence
        token_ids.append(token_id)
