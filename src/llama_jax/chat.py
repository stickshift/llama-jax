from collections.abc import Iterator, Mapping, Sequence
import logging
from typing import Annotated, Any, Callable, NamedTuple, cast
from uuid import uuid4

from jax import Array, random
from jax import numpy as jnp
from pydantic import BeforeValidator, TypeAdapter

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, TrainingLevel
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "CompletionEvent",
    "Message",
    "MessageLike",
    "Thread",
    "ThreadLike",
    "generator",
    "load_threads",
]

logger = logging.getLogger(__name__)


class Message(NamedTuple):
    """Message in a conversation."""

    role: str

    content: str


MessageLike = Mapping[str, str] | Message


class Thread(NamedTuple):
    """Multiple messages strong together in a dialog."""

    id: str

    messages: Sequence[Message]


def _validate_thread(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    if "id" not in data:
        data["id"] = uuid4().hex

    return data


Thread = Annotated[Thread, BeforeValidator(_validate_thread)]  # type: ignore


ThreadLike = Mapping[str, str] | Thread


class CompletionEvent(NamedTuple):
    """Chat completion event."""

    thread: Thread
    delta: Message | None


ChatGenerator = Callable[[ThreadLike | Sequence[ThreadLike]], Iterator[CompletionEvent | Sequence[CompletionEvent]]]

ROLE = "assistant"


def generator(
    config: ModelConfig,
    *,
    model: Model | None = None,
    key: Array | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> ChatGenerator:
    """Create a chat generator."""
    # Defaults
    max_tokens = default_arg(max_tokens, 32)
    stream = default_arg(stream, False)

    # Validate
    if config.training is not TrainingLevel.INSTRUCT:
        raise ValueError("Chat generator requires INSTRUCT model.")

    # Initialize tokenizer
    tokenizer = ll.checkpoint.load_tokenizer(config)

    # Initialize model if not provided
    if model is None:
        model = ll.model.create(config, ll.checkpoint.load_parameters(config))

    # Define generator callable
    def wrapper(
        input_threads: ThreadLike | Sequence[ThreadLike],
        **kwargs: Any,
    ) -> Iterator[CompletionEvent | Sequence[CompletionEvent]]:
        nonlocal key, max_tokens, stream

        # Override ctor args
        max_tokens = kwargs.get("max_tokens", max_tokens)
        assert max_tokens is not None
        stream = kwargs.get("stream", stream)
        assert stream is not None

        # Validate
        batched = _batched(input_threads)
        threads = load_threads(input_threads)
        bs = len(threads)

        # Render prompts and tokenize
        prompts = tuple(render_prompt(thread) for thread in threads)
        token_ids, position_mask = tokenizer.encode(prompts)

        # Initialize key/value cache
        kv_cache = ll.kv_cache.create(config, bs=token_ids.shape[0])

        # Initialize x with entire sequence on first pass
        x = token_ids

        # All sequences in batch start off active
        active = jnp.ones(x.shape[0], dtype=bool)

        content = ["" for _ in range(bs)]

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

            # Decode next token
            tokens = tokenizer.decode(next_token_id, special=False)

            # Collect tokens for active sequences
            for i in range(bs):
                if active[i]:
                    content[i] += tokens[i]

            if stream:
                events = tuple(
                    CompletionEvent(
                        thread=_response_thread(threads[i], content[i]),
                        delta=_delta_message(tokens[i]),
                    )
                    for i in range(bs)
                    if active[i]
                )

                yield events if batched else events[0]

            # Subsequent iterations process one token at a time.
            x = next_token_id
            position_mask = ll.model.increment_position_mask(position_mask)

        # Final events
        events = tuple(
            CompletionEvent(
                thread=_response_thread(threads[i], content[i]),
                delta=None,
            )
            for i in range(bs)
        )

        yield events if batched else events[0]

    return wrapper


def render_prompt(thread: Thread) -> str:
    """Render messages as Llama prompt."""
    prompt = ""

    for message in thread.messages:
        prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
        prompt += message.content
        prompt += "<|eot_id|>\n"

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def _batched(input_threads: ThreadLike | Sequence[ThreadLike]) -> bool:
    return isinstance(input_threads, Sequence)


def load_threads(input_threads: ThreadLike | Sequence[ThreadLike]) -> Sequence[Thread]:
    """Load threads."""
    if not _batched(input_threads):
        input_threads = cast(Sequence[ThreadLike], [input_threads])

    threads: Sequence[Thread] = TypeAdapter(Sequence[Thread]).validate_python(input_threads)

    return threads


def _response_thread(thread: Thread, content: str) -> Thread:
    messages = (*thread.messages, Message(role=ROLE, content=content))

    return thread._replace(messages=messages)


def _delta_message(token: str) -> Message:
    return Message(role=ROLE, content=token)
