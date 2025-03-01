from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns as timer

from rich.console import RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

import llama_jax as ll

__all__ = [
    "TokenView",
    "token_view",
]


class TokenView:
    """Supports live view of token generation."""

    def __init__(self, live: Live):
        self.live = live
        self.content = ""
        self.count = 0
        self.start_time = timer()

    def add_token(self, token: str) -> None:
        """Appends token to existing view."""
        self.content += token
        self.count += 1

        duration = (timer() - self.start_time) / 1000000000
        tps = self.count / duration

        self.live.update(_render(self.content, self.count, duration, tps))


@contextmanager
def token_view() -> Iterator[TokenView]:
    """Manages live view of token generation."""
    with Live(console=ll.tools.console()) as live:
        yield TokenView(live)


def _render(content: str, count: int, duration: float, tps: float) -> RenderableType:
    """Render content as Rich renderable."""
    footer_style = "dim"

    table = Table(
        show_header=False,
        show_edge=False,
        border_style=footer_style,
        expand=True,
    )

    # Body
    table.add_row(Markdown(content))
    table.add_row()

    # Footer
    table.add_section()
    table.add_row(f"tokens: {count}, duration: {duration:0.2f}, tps: {tps:0.2f}", style=footer_style)

    return table
