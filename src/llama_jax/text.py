"""Text completions."""

from collections.abc import Iterator

from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "generate",
]


def generate(
    key: ArrayLike,
    config: ModelConfig,
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

    # Load tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Split prompt into mutable list of tokens
    token_ids = tokenizer.encode(prompt).tolist()

    # Generate output until we get a stop token or we exceed max_tokens.
    for _ in range(max_tokens):
        # Load token ids into array
        x = jnp.array(token_ids)

        # Transform token ids into next token logits
        logits = ll.model.forward(model, x)

        # Sample tokens
        key, subkey = random.split(key)
        token_id = ll.head.sample_token(subkey, logits)

        # Check stopping criteria
        if token_id in tokenizer.stop_tokens:
            break

        # Yield token
        yield tokenizer.decode(jnp.array([token_id]))

        # Append to end of sequence
        token_ids.append(token_id)
