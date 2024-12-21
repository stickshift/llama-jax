from collections.abc import Mapping, Sequence
from functools import partial
from typing import Callable, NamedTuple

from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, TrainingLevel
from llama_jax.model import Model
from llama_jax.tokenizer import Tokenizer
from llama_jax.tools import default_arg

__all__ = [
    "Message",
    "MessageLike",
    "generator",
    "render_prompt",
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
    config: ModelConfig,
    model: Model | None = None,
    key: ArrayLike | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> Callable[[Sequence[MessageLike]], CompletionResponse]:
    """Create a chat generator."""
    # Defaults
    key = default_arg(key, default_factory=partial(random.key, 42))
    max_tokens = default_arg(max_tokens, 32)

    # Initialize tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Initialize model if not provided
    if model is None:
        params = ll.checkpoint.load_parameters(config)
        model = ll.model.create(config, params)

    return partial(
        _generate,
        config=config,
        tokenizer=tokenizer,
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
    config: ModelConfig,
    tokenizer: Tokenizer,
    model: Model,
    key: ArrayLike,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_tokens: int,
) -> CompletionResponse:
    """Generate next response in conversation."""
    # Render prompt
    prompt = render_prompt(config, messages)

    # Split prompt into tokens
    token_ids = tokenizer.encode(prompt)

    # Convert token ids into mutable list
    token_ids = token_ids.tolist()

    content = ""

    # Generate output until we get a stop token or we exceed max_tokens.
    for _ in range(max_tokens):
        # Initialize x with current token ids
        x = jnp.array(token_ids)

        # Stack token ids into batch size of 1
        x = jnp.reshape(x, (1,) + x.shape)

        # Transform token ids into next token logits
        output = ll.model.forward(config, model, x)

        # Sample tokens
        token_id, key = ll.head.sample_token(
            output.logits,
            key=key,
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
        token_ids.append(token_id[0])

    message = Message(role="assistant", content=content)

    return CompletionResponse(messages=(*messages, message))


def render_prompt(config: ModelConfig, messages: Sequence[MessageLike]) -> str:
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
