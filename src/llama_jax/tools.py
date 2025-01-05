from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import Callable, cast, no_type_check

__all__ = [
    "default_arg",
    "executor",
    "recursive_tuple",
    "shell",
]


def default_arg[T](
    v: T | None,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
) -> T:
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return cast(T, default)


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
    global _executor  # noqa: PLW0603
    if _executor is None:
        _executor = ThreadPoolExecutor()
    return _executor


@no_type_check
def recursive_tuple(x):
    """Convert list to tuple recursively."""
    return tuple(recursive_tuple(v) if isinstance(v, list) else v for v in x)
