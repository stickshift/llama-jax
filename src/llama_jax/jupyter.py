from collections.abc import Callable
from time import perf_counter_ns as seed
from typing import Any, NamedTuple

from IPython.core.magic import register_line_cell_magic
from jax import random

import llama_jax as ll
from llama_jax.chat import ChatSession
from llama_jax.checkpoint import ModelConfig
from llama_jax.tools import default_arg

__all__ = [
    "JupyterSession",
    "session",
]


# class JupyterSession(NamedTuple):
#     """Configure notebook-wide chat session and magics."""

#     chat: ChatSession

#     cell_magic: Callable[[str, str | None], None]


_system_prompt = ll.tools.prompt(
    """
    You are an expert data scientist. Your job is to help me analyze data in a Jupyter notebook.

    - All of your responses should be formatted as markdown.
    - Answer in a single sentence whenever possible or not otherwise instructed.
    """
)


def session(
    config: ModelConfig | None = None,
    system_prompt: str | None = None,
    warmup: bool | None = None,
    warmup_tokens: int | None = None,
    **kwargs: Any,
) -> ChatSession:
    """Start a new Jupyter session."""
    # Defaults
    warmup = default_arg(warmup, True)
    system_prompt = default_arg(system_prompt, _system_prompt)

    # Initialize chat
    session = ll.chat.session(
        config=config,
        system_prompt=system_prompt,
        warmup=warmup,
        warmup_tokens=warmup_tokens,
        **kwargs,
    )

    # Register chat magic
    @register_line_cell_magic  # type: ignore[misc]
    def chat(line: str, cell: str | None = None) -> None:
        content = default_arg(cell, line)
        _submit_chat(session, content)

    return session


def _submit_chat(session: ChatSession, content: str) -> None:
    """Submit new chat request."""
    key = random.key(seed())

    with ll.render.token_view(session.config, prompt=content) as view:
        for token in ll.chat.complete(session, content=content, key=key):
            view.add_token(token)
