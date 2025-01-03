from jax import Array, random, numpy as jnp
import pytest

__all__ = [
    "key",
    "assert_similar_arrays",
]


@pytest.fixture
def key() -> Array:
    """Random number generator key."""
    return random.key(42)


def assert_similar_arrays(x: Array, y: Array):
    """Asserts vectors along last dimension of x and y are similar."""

    # Sanity check
    assert x.shape == y.shape

    # Reshape x and y to (..., n)
    x = jnp.reshape(x, (-1, x.shape[-1]))
    y = jnp.reshape(y, (-1, y.shape[-1]))
    n = x.shape[0]

    # Compare x and y using cosine similarity
    scores = jnp.array([jnp.dot(x[i], y[i]) / jnp.sum(jnp.pow(y[i], 2)) for i in range(n)])

    # Verify scores are all close to perfect 1.0
    assert jnp.allclose(scores, jnp.ones(n))
