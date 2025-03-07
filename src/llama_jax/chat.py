from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
import logging
from queue import SimpleQueue
from random import sample
from threading import Event
from time import time_ns as seed
from typing import Any, NamedTuple
from uuid import uuid4

from jax import Array, random
from jax import numpy as jnp
from pydantic import TypeAdapter
from tqdm.auto import tqdm

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


class Completion(NamedTuple):
    messages: Sequence[Message]

    delta: str | None


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


class ChatSession(NamedTuple):
    """Chat session state."""

    id: str

    config: ModelConfig

    tokenizer: Tokenizer

    model: Model

    system_prompt: str | None

    executor: ThreadPoolExecutor


_default_checkpoint = "Llama3.2-3B-Instruct"


def session(
    config: ModelConfig | None = None,
    system_prompt: str | None = None,
    warmup_tokens: int | None = None,
    **kwargs: Any,
) -> ChatSession:
    """Create new chat session."""
    # Load config if needed
    if config is None:
        max_sequence_length = kwargs.pop("max_sequence_length", 4096)
        config = ll.checkpoint.load_config(_default_checkpoint, max_sequence_length=max_sequence_length, **kwargs)

    # Validate
    if not config.checkpoint_name.endswith("-Instruct"):
        raise ValueError(f"Invalid checkpoint {config.checkpoint_name}. Chat sessions require instruct checkpoint.")

    session = ChatSession(
        id=uuid4().hex,
        config=config,
        tokenizer=ll.checkpoint.load_tokenizer(config),
        model=ll.model.create(config),
        system_prompt=system_prompt,
        executor=ThreadPoolExecutor(),
    )

    # Warmup model
    if warmup_tokens:
        _warmup(session, max_tokens=warmup_tokens)

    return session


def complete(
    session: ChatSession,
    *,
    messages: Sequence[MessageLike],
    stream: bool | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Completion | Iterator[Completion]:
    """Complete chat."""
    # Defaults
    stream = default_arg(stream, False)

    completion_generator = _complete(
        session,
        input_messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if stream:
        return completion_generator

    # Collect completion events
    completion = next(c for c in completion_generator if c.delta is None)

    return completion


_token_id_timeout = None


def _complete(
    session: ChatSession,
    *,
    input_messages: Sequence[MessageLike],
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Iterator[Completion]:
    """Completion generator."""
    q = SimpleQueue[Array | None]()
    cancel_event = Event()

    # Prompt
    messages = load_messages(input_messages)
    if not messages:
        raise ValueError("input_messages must include at least 1 message")

    if session.system_prompt and messages[0].role != "system":
        messages = [Message(role="system", content=session.system_prompt), *messages]

    prompt = render_prompt(messages)

    # Encode token ids
    token_ids, position_mask = session.tokenizer.encode(prompt)

    # Schedule generator
    task = session.executor.submit(
        _generator,
        session,
        q,
        cancel_event,
        token_ids,
        position_mask,
        max_tokens,
        temperature,
        top_k,
        top_p,
    )

    try:
        # Collect generated tokens
        content = ""
        while (token_id := q.get(timeout=_token_id_timeout)) is not None:
            # Decode token ids
            token = session.tokenizer.decode(token_id)[0]
            content += token

            yield Completion(
                messages=[*messages, Message(role="assistant", content=content)],
                delta=token,
            )
    finally:
        _cancel_task(task, cancel_event)

    # Final completion
    yield Completion(
        messages=[*messages, Message(role="assistant", content=content)],
        delta=None,
    )


def _generator(  # noqa: PLR0917
    session: ChatSession,
    q: SimpleQueue[Array | None],
    cancel_event: Event,
    token_ids: Array,
    position_mask: Array,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> None:
    """Generate up to max_tokens."""
    job_id = uuid4().hex

    # Defaults
    n = token_ids.shape[-1]
    max_tokens = default_arg(max_tokens, session.config.max_sequence_length - n)

    x = token_ids
    kvc = ll.kvc.create(session.config)
    keys = random.split(random.key(seed()), max_tokens)

    logger.info(f"Generator: {job_id} - started")

    try:
        for i in range(max_tokens):
            # Check for cancel
            if cancel_event.is_set():
                break

            # Predict next token
            logits, kvc = ll.model.forward(session.config, session.model, x, position_mask, kvc=kvc)
            token_id = ll.model.next_token_id(
                logits,
                key=keys[i],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Check for stop tokens
            if jnp.any(jnp.isin(token_id.squeeze(), session.tokenizer.stop_tokens)):
                break

            # Publish next token id
            q.put(token_id)

            x = token_id

        logger.info(f"Generator: {job_id} - completed")

    except Exception as e:
        logger.info(f"Generator: {job_id} - failed - {e}")
        raise e

    finally:
        # Publish done
        q.put(None)


def _cancel_task(task: Future[None], cancel_event: Event) -> None:
    # Tell background thread to quit
    cancel_event.set()

    # Wait for cancel
    done, _ = wait([task])
    assert task in done


_warmup_prompt_pool = (
    "Write a short poem about the beauty of nature",
    "Explain the concept of artificial intelligence to a 10-year-old",
    "Summarize the main points of the book 'To Kill a Mockingbird'",
    "Generate a short story about a character who discovers a hidden world",
    "Translate the phrase 'Hello, how are you?' into French",
    "Describe the process of photosynthesis in simple terms",
    "Create a new language with its own grammar and syntax",
    "Explain the difference between a hypothesis and a theory",
    "Write a script for a conversation between a human and a chatbot",
    "Summarize the main points of the movie 'The Shawshank Redemption'",
    "Generate a list of 10 synonyms for the word 'happy'",
    "Explain the concept of blockchain technology in simple terms",
    "Create a new species of animal with its own characteristics",
    "Describe the process of mitosis in simple terms",
    "Write a short story about a character who travels back in time",
    "Summarize the main points of the book '1984'",
    "Generate a poem about the beauty of the ocean",
    "Explain the concept of quantum physics to a 15-year-old",
    "Create a new sport with its own rules and equipment",
    "Describe the process of cellular respiration in simple terms",
    "Write a script for a conversation between two historical figures",
)


def _warmup(session: ChatSession, max_tokens: int | None = None) -> None:
    """Warmup model cache."""
    max_tokens = default_arg(max_tokens, session.config.max_sequence_length)

    n = 10
    prompts = sample(_warmup_prompt_pool, k=n)

    for i, prompt in enumerate(prompts):
        generator = complete(
            session,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=max_tokens,
        )
        for _ in tqdm(generator, desc=f"Warmup ({i + 1}/{n})", total=max_tokens, leave=False):
            pass
