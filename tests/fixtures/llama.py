import pytest

__all__ = [
    "bs",
    "n",
]


@pytest.fixture
def bs() -> int:
    """Batch size."""
    return 2


@pytest.fixture
def n() -> int:
    """Sequence length."""
    return 10
