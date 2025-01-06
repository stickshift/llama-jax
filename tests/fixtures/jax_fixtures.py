from time import time_ns

from jax import Array, random
from jax import numpy as jnp
import pytest

from llama_jax.tools import default_arg

__all__ = [
    "assert_similar_arrays",
    "key",
]


@pytest.fixture
def key() -> Array:
    """Random number generator key."""
    return random.key(time_ns())


def similarity_scores(x: Array, y: Array):
    """Compares vectors along last dimension of x and y are similar."""

    # Sanity check
    assert x.shape == y.shape

    # Reshape x and y to (..., n)
    x = jnp.reshape(x, (-1, x.shape[-1]))
    y = jnp.reshape(y, (-1, y.shape[-1]))
    n = x.shape[0]

    # Compare x and y using cosine similarity
    scores = jnp.array([jnp.dot(x[i], y[i]) / jnp.sum(jnp.pow(y[i], 2)) for i in range(n)])

    return scores, n


def assert_similar_arrays(x: Array, y: Array, atol: float | None = None):
    """Asserts vectors along last dimension of x and y are similar."""

    # Defaults
    atol = default_arg(atol, 0.03)

    # Compare x and y using cosine similarity
    scores, n = similarity_scores(x, y)

    # Verify scores are all close to perfect 1.0
    assert jnp.allclose(scores, jnp.ones(n), atol=atol)
