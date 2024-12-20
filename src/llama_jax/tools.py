from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import Callable

__all__ = [
    "default_arg",
    "shell",
    "executor",
]


def default_arg[T](
    v: T,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
):
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return default


def shell(command: str) -> str:
    """Run shell command."""
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return result.stdout.strip()


_executor = None

def executor() -> ThreadPoolExecutor:
    """Global executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor()
    return _executor