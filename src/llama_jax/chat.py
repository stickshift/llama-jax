from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
import logging
from queue import Queue
from sys import stdout
from time import perf_counter_ns as seed
from typing import Any, NamedTuple
from uuid import uuid4

from jax import Array, random
from jax import numpy as jnp
from pydantic import TypeAdapter

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig
from llama_jax.model import Model
from llama_jax.tokenizer import Tokenizer
from llama_jax.tools import default_arg

__all__ = [
    "ChatSession",
    "Message",
    "MessageLike",
    "complete",
    "load_messages",
    "render_prompt",
    "session",
]

logger = logging.getLogger(__name__)


class Message(NamedTuple):
    """Message in a conversation."""

    role: str

    content: str


MessageLike = Mapping[str, str] | Message


def load_messages(input_messages: MessageLike | Sequence[MessageLike]) -> Sequence[Message]:
    """Load messages."""
    if not isinstance(input_messages, Sequence):
        input_messages = [input_messages]

    return TypeAdapter(Sequence[Message]).validate_python(input_messages)


def render_prompt(messages: Sequence[Message]) -> str:
    """Render messages as Llama prompt."""
    prompt = ""

    for message in messages:
        prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
        prompt += message.content
        prompt += "<|eot_id|>\n"

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


_messages: Mapping[str, list[Message]] = {}


class ChatSession(NamedTuple):
    """Chat session state."""

    id: str

    config: ModelConfig

    tokenizer: Tokenizer

    model: Model

    executor: ThreadPoolExecutor


_default_checkpoint = "Llama3.2-3B-Instruct"


def session(
    config: ModelConfig | None = None,
    system_prompt: str | None = None,
    warmup: bool | None = None,
    warmup_tokens: int | None = None,
    **kwargs: Any,
) -> ChatSession:
    """Create new chat session."""
    # Defaults
    if config is None:
        config = ll.checkpoint.load_config(_default_checkpoint, **kwargs)

    # Validate
    if not config.checkpoint_name.endswith("-Instruct"):
        raise ValueError(f"Invalid checkpoint {config.checkpoint_name}. Chat sessions require instruct checkpoint.")

    id = uuid4().hex
    tokenizer = ll.checkpoint.load_tokenizer(config)
    params = ll.checkpoint.load_parameters(config)
    model = ll.model.create(config, params)

    # Initialize executor with max of 1 worker to ensure multiple requests are queued
    executor = ThreadPoolExecutor(max_workers=1)

    # Initialize conversation
    _messages[id] = [Message(role="system", content=system_prompt)] if system_prompt else []  # type: ignore[index]

    session = ChatSession(id=id, config=config, tokenizer=tokenizer, model=model, executor=executor)

    # Warmup model
    if warmup:
        _warmup(session, random.key(seed()), max_tokens=warmup_tokens)

    return session


def _warmup(session: ChatSession, key: Array, max_tokens: int | None = None) -> None:
    """Warmup model cache."""
    max_tokens = default_arg(max_tokens, session.config.max_tokens)

    prompt = render_prompt([
        Message(
            role="user",
            content="Generate 2 sentences of lorem ipsum",
        )
    ])

    tokenizer = ll.checkpoint.load_tokenizer(session.config)
    token_ids, position_mask = tokenizer.encode(prompt)

    # q is unbounded to avoid blocking
    q = Queue[Array | None]()

    # Feed model in background
    job = session.executor.submit(
        _generate_tokens,
        session,
        token_ids,
        position_mask,
        key,
        q,
        max_tokens=max_tokens,
    )

    # Stream tokens from queue
    while True:
        next_token_id = q.get()
        if next_token_id is None:
            q.task_done()
            break

        stdout.write(".")

        q.task_done()

    logger.info("Waiting for background job to finish")
    wait([job])

    logger.info("Warmup complete")


def complete(session: ChatSession, *, content: str, key: Array) -> Iterator[str]:
    """Submit chat completion."""
    # q is unbounded to avoid blocking
    q = Queue[Array | None]()

    # Append to existing conversation
    _messages[session.id].append(
        Message(
            role="user",
            content=content,
        )
    )

    prompt = render_prompt(_messages[session.id])
    token_ids, position_mask = session.tokenizer.encode(prompt)

    # Feed model in background
    job = session.executor.submit(
        _generate_tokens,
        session,
        token_ids,
        position_mask,
        key,
        q,
    )

    # Stream tokens from queue
    response = ""
    while True:
        next_token_id = q.get()
        if next_token_id is None:
            q.task_done()
            break

        token = session.tokenizer.decode(next_token_id)[0]

        yield token

        response += token

        q.task_done()

    # Append response to existing conversation
    _messages[session.id].append(
        Message(
            role="assistant",
            content=response,
        )
    )

    logger.info("Waiting for background job to finish")
    wait([job])

    logger.info("Foreground job complete")


def _generate_tokens(
    session: ChatSession,
    token_ids: Array,
    position_mask: Array,
    key: Array,
    q: Queue[Array | None],
    max_tokens: int | None = None,
) -> None:
    """Feed model in background."""
    # Defaults
    max_tokens = default_arg(max_tokens, session.config.max_tokens)

    config, tokenizer, model = session.config, session.tokenizer, session.model
    bs, n = token_ids.shape

    # Generate up to config.max_tokens
    max_tokens = min(max_tokens, config.max_tokens - n)
    logger.info(f"Background: started - generating up to {max_tokens} tokens")

    x = token_ids
    kvc = ll.kvc.create(config)
    key, *subkeys = random.split(key, max_tokens + 1)

    # All sequences in batch start off active
    active = jnp.ones(bs, dtype=bool)

    for i in range(max_tokens):
        # Transform x into next token id
        logits, kvc = ll.model.forward(config, model, x, position_mask, kvc=kvc)
        next_token_id = ll.model.next_token(logits, key=subkeys[i])

        # Track active sequences
        is_stop_token = jnp.isin(next_token_id.squeeze(), tokenizer.stop_tokens)
        active = active & ~is_stop_token
        if not jnp.any(active):
            break

        # Publish next token id
        q.put(next_token_id)

        x = next_token_id

    # Send all done
    q.put(None)

    # Wait for all jobs to be processed
    q.join()

    logger.info("Background: completed")
