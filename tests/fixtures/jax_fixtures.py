from jax import Array, random
import pytest

__all__ = [
    "key",
]


@pytest.fixture
def key() -> Array:
    """Random number generator key."""
    return random.key(42)
