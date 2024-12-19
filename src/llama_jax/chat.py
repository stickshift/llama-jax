from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .model import Model

__all__ = [
    "completion",
]


class Message(NamedTuple):
    """Message in a conversation."""

    role: str

    content: str


MessageLike = Mapping[str, str] | Message


class CompletionResponse(NamedTuple):
    """Completion response."""

    messages: Sequence[Message]


def completion(model: Model, messages: Sequence[MessageLike]) -> CompletionResponse:
    """Generate chat completion."""
    pass
