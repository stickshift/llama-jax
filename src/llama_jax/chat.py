from collections.abc import Mapping, Sequence
from functools import partial
from typing import Callable, NamedTuple

from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, TrainingLevel
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "generator",
]


class Message(NamedTuple):
    """Message in a conversation."""

    role: str

    content: str


MessageLike = Mapping[str, str] | Message


class CompletionResponse(NamedTuple):
    """Completion response."""

    messages: Sequence[Message]


def generator(
    model: Model,
    key: ArrayLike | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> Callable[[Sequence[MessageLike]], CompletionResponse]:
    """Create a chat generator."""
    # Defaults
    key = default_arg(key, default_factory=random.key)
    max_tokens = default_arg(max_tokens, 32)

    return partial(
        _generate,
        model=model,
        key=key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _generate(
    messages: Sequence[MessageLike],
    *,
    model: Model,
    key: ArrayLike,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_tokens: int,
) -> CompletionResponse:
    """Generate tokens given a prompt."""
    # Render prompt
    prompt = _render_prompt(model.config, messages)

    # Split prompt into tokens
    tokenizer = ll.checkpoint.load_tokenizer(model.config)
    token_ids = tokenizer.encode(prompt)

    # Convert token ids into mutable list
    token_ids = token_ids.tolist()

    content = ""

    # Generate output until we get a stop token or we exceed max_tokens.
    for _ in range(max_tokens):
        # Initialize x with current token ids
        x = jnp.array(token_ids)

        # Transform token ids into next token logits
        logits = ll.model.forward(model, x)

        # Sample tokens
        key, subkey = random.split(key)
        token_id = ll.head.sample_token(
            logits,
            key=subkey,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Check stopping criteria
        if token_id in tokenizer.stop_tokens:
            break

        # Decode next token
        content += tokenizer.decode(token_id)

        # Append to end of sequence
        token_ids.append(token_id)

    message = Message(role="assistant", content=content)

    return CompletionResponse(messages=(*messages, message))


def _render_prompt(config: ModelConfig, messages: Sequence[MessageLike]) -> str:
    """Render messages."""
    prompt = ""

    for data in messages:
        message = data

        # Convert dicts to Messages
        if isinstance(message, dict):
            message = Message(**message)

        if config.training == TrainingLevel.INSTRUCT:
            prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"

        prompt += message.content

        if config.training == TrainingLevel.PRETRAINED:
            prompt += "\n\n"

        if config.training == TrainingLevel.INSTRUCT:
            prompt += "<|eot_id|>"

    if config.training == TrainingLevel.INSTRUCT:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt
