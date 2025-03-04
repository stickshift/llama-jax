from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
import logging
from queue import Queue
from random import sample
from threading import Event
from time import perf_counter_ns as seed
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
    warmup = default_arg(warmup, True)

    # Load config if needed
    if config is None:
        max_tokens = kwargs.pop("max_tokens", 4096)
        config = ll.checkpoint.load_config(_default_checkpoint, max_tokens=max_tokens, **kwargs)

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


def complete(session: ChatSession, *, content: str, key: Array, max_tokens: int | None = None) -> Iterator[str]:
    """Submit chat completion."""
    logger.info("Completion: started")

    # q is unbounded to avoid blocking
    q = Queue[Array | None]()

    # Append to existing conversation
    messages = _messages[session.id] + [
        Message(
            role="user",
            content=content,
        ),
    ]

    #
    # TODO: Revisit this with more time to optimize it.
    #
    # _messages[session.id].append(
    #     Message(
    #         role="user",
    #         content=content,
    #     )
    # )

    prompt = render_prompt(messages)
    token_ids, position_mask = session.tokenizer.encode(prompt)

    logger.info(f"Completion: split prompt into {token_ids.shape[-1]} token ids")

    # Schedule generator in background thread
    cancel_event = Event()
    job = session.executor.submit(
        _generator,
        session,
        token_ids,
        position_mask,
        key,
        q,
        cancel_event,
        max_tokens,
    )

    logger.info("Completion: scheduled generator")

    try:
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

        #
        # TODO: Revisit this with more time to optimize it.
        #
        # Append response to existing conversation
        # _messages[session.id].append(
        #     Message(
        #         role="assistant",
        #         content=response,
        #     )
        # )

    except KeyboardInterrupt:
        # Cancel job
        cancel_event.set()
        raise

    logger.info("Completion: waiting for generator to finish")

    wait([job])

    logger.info("Completion: completed")


def _generator(
    session: ChatSession,
    token_ids: Array,
    position_mask: Array,
    key: Array,
    q: Queue[Array | None],
    cancel_event: Event,
    max_tokens: int | None = None,
) -> None:
    """Background job that feeds tokens into model."""
    # Defaults
    max_tokens = default_arg(max_tokens, session.config.max_sequence_length)

    config, tokenizer, model = session.config, session.tokenizer, session.model
    bs, n = token_ids.shape

    # Generate up to config.max_sequence_length
    max_tokens = min(max_tokens, config.max_sequence_length - n)
    logger.info(f"Generator: started - generating up to {max_tokens} tokens")

    x = token_ids
    kvc = ll.kvc.create(config)
    key, *subkeys = random.split(key, max_tokens + 1)

    # All sequences in batch start off active
    active = jnp.ones(bs, dtype=bool)

    try:
        # Generate up to max tokens
        for i in range(max_tokens):
            # Check for cancel
            if cancel_event.is_set():
                logger.info("Generator: cancelled")
                return

            # Transform token ids
            logits, kvc = ll.model.forward(config, model, x, position_mask, kvc=kvc)

            # Sample tokens
            next_token_id = ll.model.next_token(logits, key=subkeys[i])

            # Track active sequences
            is_stop_token = jnp.isin(next_token_id.squeeze(), tokenizer.stop_tokens)
            active = active & ~is_stop_token
            if not jnp.any(active):
                break

            # Publish next token id
            q.put(next_token_id)

            # Process generated token on next pass
            x = next_token_id

        # Send all done
        q.put(None)

        # Wait for all jobs to be processed
        q.join()

        logger.info("Generator: completed")

    except Exception as e:
        logger.exception(f"Generator: failed - {e}")
        raise e


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


def _warmup(session: ChatSession, key: Array, max_tokens: int | None = None) -> None:
    """Warmup model cache."""
    max_tokens = default_arg(max_tokens, session.config.max_sequence_length)

    n = 10
    prompts = sample(_warmup_prompt_pool, k=n)
    key, *subkeys = random.split(key, n + 1)

    for i, prompt in enumerate(prompts):
        generator = complete(session, content=prompt, key=subkeys[i], max_tokens=max_tokens)
        for _ in tqdm(generator, desc=f"Warmup ({i + 1}/{n})", total=max_tokens, leave=False):
            pass
