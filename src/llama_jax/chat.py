from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import Any, Callable, NamedTuple, cast

from jax import Array, random

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig, TrainingLevel
from llama_jax.model import Model
from llama_jax.tools import default_arg

__all__ = [
    "CompletionEvent",
    "Message",
    "MessageLike",
    "generator",
]


class Message(NamedTuple):
    """Message in a conversation."""

    role: str

    content: str


MessageLike = Mapping[str, str] | Message


class CompletionEvent(NamedTuple):
    """Chat completion event."""

    messages: Sequence[Message]
    delta: Message | None


ChatGenerator = Callable[[Sequence[MessageLike]], Iterator[CompletionEvent]]

ROLE = "assistant"


def generator(
    config: ModelConfig,
    key: Array,
    *,
    model: Model | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> tuple[ChatGenerator, Array]:
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
        input_messages: Sequence[MessageLike],
        *,
        key: Array,
        **kwargs: Any,
    ) -> Iterator[CompletionEvent]:
        # Override ctor args
        nonlocal max_tokens, stream
        max_tokens = kwargs.get("max_tokens", max_tokens)
        assert max_tokens is not None
        stream = kwargs.get("stream", stream)
        assert stream is not None

        # Validate input messages
        messages = _validate_messages(input_messages)

        # Render prompt and split into token ids
        prompt = render_prompt(messages)
        token_ids = tokenizer.encode(prompt)

        # Initialize key/value cache
        kv_cache = ll.kv_cache.create(config)

        # Initialize x with entire sequence on first pass
        x = token_ids

        content = ""

        # Sample up to max tokens
        for _ in range(max_tokens):
            # Transform x into logits
            logits, kv_cache = ll.model.forward(config, model, x, kv_cache=kv_cache)

            # Sample next token
            next_token_id, key = ll.model.next_token(logits, key, temperature=temperature, top_k=top_k, top_p=top_p)

            # Break on stop tokens
            if next_token_id in tokenizer.stop_tokens:
                break

            # Decode next token
            token = tokenizer.decode(next_token_id)[0]

            # Append to final response
            content += token

            if stream:
                yield CompletionEvent(
                    messages=_response_messages(messages, content),
                    delta=_delta_message(token),
                )

            # Subsequent iterations process one token at a time
            x = next_token_id

        # Final event
        yield CompletionEvent(messages=_response_messages(messages, content), delta=None)

    # Generate subkey to be consumed by wrapper
    key, subkey = random.split(key)
    wrapper = partial(wrapper, key=subkey)

    return wrapper, key


def render_prompt(messages: Sequence[Message]) -> str:
    """Render messages as Llama prompt."""
    prompt = ""

    for message in messages:
        prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
        prompt += message.content
        prompt += "<|eot_id|>\n"

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def _validate_messages(messages: Sequence[MessageLike]) -> Sequence[Message]:
    validated = tuple(Message(**m) if isinstance(m, dict) else m for m in messages)
    return cast(Sequence[Message], validated)


def _response_messages(messages: Sequence[Message], content: str) -> Sequence[Message]:
    return *messages, Message(role=ROLE, content=content)


def _delta_message(token: str) -> Message:
    return Message(role=ROLE, content=token)


#
#
# def generator(
#     config: ModelConfig,
#     model: Model | None = None,
#     key: Array | None = None,
#     temperature: float | None = None,
#     top_k: int | None = None,
#     top_p: float | None = None,
#     max_tokens: int | None = None,
# ) -> Callable[[Sequence[MessageLike]], CompletionResponse]:
#     """Create a chat generator."""
#     # Defaults
#     key = default_arg(key, default_factory=partial(random.key, 42))
#     max_tokens = default_arg(max_tokens, 32)
#
#     # Initialize tokenizer
#     tokenizer = ll.checkpoint.load_tokenizer(config)
#
#     # Initialize model if not provided
#     if model is None:
#         params = ll.checkpoint.load_parameters(config)
#         model = ll.model.create(config, params)
#
#     return partial(
#         _generate,
#         config=config,
#         tokenizer=tokenizer,
#         model=model,
#         key=key,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p,
#         max_tokens=max_tokens,
#     )
#
#
# def _generate(
#     messages: Sequence[MessageLike],
#     *,
#     config: ModelConfig,
#     tokenizer: Tokenizer,
#     model: Model,
#     key: Array,
#     temperature: float | None,
#     top_k: int | None,
#     top_p: float | None,
#     max_tokens: int,
# ) -> CompletionResponse:
#     """Generate next response in conversation."""
#     # Render prompt
#     prompt = render_prompt(config, messages)
#
#     # Split prompt into tokens
#     token_ids = tokenizer.encode(prompt)
#
#     # Convert token ids into mutable list
#     token_ids = token_ids.tolist()
#
#     content = ""
#
#     # Generate output until we get a stop token or we exceed max_tokens.
#     for _ in range(max_tokens):
#         # Initialize x with current token ids
#         x = jnp.array(token_ids)
#
#         # Stack token ids into batch size of 1
#         x = jnp.reshape(x, (1, *x.shape))
#
#         # Transform token ids into next token logits
#         output = ll.model.forward(config, model, x)
#
#         # Sample tokens
#         token_id, key = ll.head.sample_token(
#             output.logits,
#             key=key,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#         )
#
#         # Check stopping criteria
#         if token_id in tokenizer.stop_tokens:
#             break
#
#         # Decode next token
#         content += tokenizer.decode(token_id)
#
#         # Append to end of sequence
#         token_ids.append(token_id[0])
#
#     message = Message(role="assistant", content=content)
#
#     return CompletionResponse(messages=(*messages, message))
#
#
# def render_prompt(config: ModelConfig, messages: Sequence[MessageLike]) -> str:
#     """Render messages."""
#     prompt = ""
#
#     for data in messages:
#         message = data
#
#         # Convert dicts to Messages
#         if isinstance(message, dict):
#             message = Message(**message)
#
#         if config.training == TrainingLevel.INSTRUCT:
#             prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n"
#
#         prompt += message.content
#
#         if config.training == TrainingLevel.PRETRAINED:
#             prompt += "\n\n"
#
#         if config.training == TrainingLevel.INSTRUCT:
#             prompt += "<|eot_id|>"
#
#     if config.training == TrainingLevel.INSTRUCT:
#         prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
#
#     return prompt
