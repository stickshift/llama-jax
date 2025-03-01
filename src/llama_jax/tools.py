from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import logging
from logging import Logger
import subprocess
from time import perf_counter_ns as timer
from typing import Callable, cast, no_type_check

from rich.console import Console

__all__ = [
    "default_arg",
    "executor",
    "recursive_tuple",
    "shell",
    "trace",
    "console",
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


_console = None


def console() -> Console:
    """Global console."""
    global _console  # noqa: PLW0603
    if _console is None:
        _console = Console()
    return _console


@no_type_check
def recursive_tuple(x):
    """Convert list to tuple recursively."""
    return tuple(recursive_tuple(v) if isinstance(v, list) else v for v in x)


def trace[**P, R](logger: Logger, log_level: int | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate function with time instrumentation."""
    # Defaults
    log_level = default_arg(log_level, logging.DEBUG)

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = timer()

            try:
                result = f(*args, **kwargs)
            finally:
                duration = (timer() - start_time) / 1000000
                logger.log(log_level, f"{f.__name__} took {duration:0.0f} ms")

            return result

        return wrapper

    return decorator
