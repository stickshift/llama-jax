from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

__all__ = [
    "TokenView",
    "token_view",
]


class TokenView(Table):
    def __init__(self):
        super().__init__(
            show_header=False, 
            show_edge=False,
            width=100,
        )

        self.content = ""
        self.add_column()
        self.add_row(Markdown(self.content))
    
    def add_token(self, token: str) -> None:
        self.content += token
        self.columns[0]._cells[0] = Markdown(self.content)


@asynccontextmanager
async def token_view() -> AsyncIterator[TokenView]:    
    view = TokenView()
    with Live(view):
        yield view
    