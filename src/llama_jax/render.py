from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns as timer

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.style import Style
from rich.table import Table
from rich.theme import Theme

import llama_jax as ll
from llama_jax.checkpoint import ModelConfig

__all__ = [
    "TokenView",
    "token_view",
]


_default_theme_overrides = {
    "markdown.code": Style(bold=True),
}

_console = None


def console() -> Console:
    """Global console."""
    global _console  # noqa: PLW0603
    if _console is None:
        _console = Console(theme=Theme(_default_theme_overrides))
    return _console


class TokenView:
    """Supports live view of token generation."""

    def __init__(self, live: Live, prompt: str | None, input_count: int | None):
        self.live = live
        self.prompt = prompt
        self.content = ""
        self.input_count = input_count
        self.output_count = 0
        self.duration = 0.0
        self.tps = 0.0
        self.start_time = timer()
        self.live.update(
            _render(
                self.prompt,
                self.content,
                self.input_count,
                self.output_count,
                self.duration,
                self.tps,
            )
        )

    def add_token(self, token: str) -> None:
        """Appends token to existing view."""
        self.content += token
        self.output_count += 1

        self.duration = (timer() - self.start_time) / 1000000000
        self.tps = self.output_count / self.duration

        self.live.update(
            _render(
                self.prompt,
                self.content,
                self.input_count,
                self.output_count,
                self.duration,
                self.tps,
            )
        )


@contextmanager
def token_view(config: ModelConfig, prompt: str | None = None) -> Iterator[TokenView]:
    """Manages live view of token generation."""
    input_count = None

    if prompt:
        tokenizer = ll.checkpoint.load_tokenizer(config)
        token_ids, _ = tokenizer.encode(prompt)
        input_count = token_ids.shape[-1]

    with Live(console=console()) as live:
        yield TokenView(live, prompt, input_count)


def _render(
    prompt: str | None,
    content: str,
    input_count: int | None,
    output_count: int,
    duration: float,
    tps: float,
) -> RenderableType:
    """Render content as Rich renderable."""
    footer_style = "dim"

    table = Table(
        show_header=False,
        show_edge=False,
        border_style=footer_style,
        expand=True,
    )

    # Header
    if prompt and len(prompt) > 80:
        prompt = prompt[:80] + "..."

    header = f"> {prompt}\n\n" if prompt else ""

    # Body
    table.add_row(Markdown(header + content, code_theme="default"))
    table.add_row()

    # Footer
    footer_content = f"input: {input_count}, " if input_count is not None else ""
    footer_content += f"output: {output_count}, duration: {duration:0.2f}, tps: {tps:0.2f}"

    table.add_section()
    table.add_row(footer_content, style=footer_style)

    return table
