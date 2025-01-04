from jax import Array, random
import jax.numpy as jnp
from pytest import approx

def test_array():
    #
    # Givens
    #

    # n = 10
    n = 10

    #
    # Whens
    #

    # I create array of length n
    x = jnp.arange(n)

    #
    # Thens
    #

    # x should have length n
    assert len(x) == n

    # x should have shape (n,)
    assert x.shape == (n,)


def test_choice(key: Array):
    """Verifies random.choice randomly samples from options according to provided probabilities."""

    #
    # Givens
    #

    # Number of iterations
    n = 1000

    # Pool w/ 3 options
    pool = jnp.array([0, 1, 2])

    # Sample routine
    def sample(probs: list[float], key: Array) -> tuple[dict, Array]:
        counts = {k.item(): 0 for k in pool}
        for _ in range(n):
            key, subkey = random.split(key)
            x = random.choice(subkey, pool, p=jnp.array(probs)).item()
            counts[x] = counts[x] + 1

        return counts, key

    #
    # Whens
    #

    # I sample using uniform probabilities
    counts, key = sample([0.33, 0.33, 0.33], key)

    #
    # Thens
    #

    # Counts should match probabilities
    assert counts[0] == approx(n / 3, rel=0.1)
    assert counts[1] == approx(n / 3, rel=0.1)
    assert counts[2] == approx(n / 3, rel=0.1)

    #
    # Whens
    #

    # I sample using non-standard uniform probabilities
    counts, key = sample([1.0, 1.0, 1.0], key)

    #
    # Thens
    #

    # Counts should match probabilities
    assert counts[0] == approx(n / 3, rel=0.1)
    assert counts[1] == approx(n / 3, rel=0.1)
    assert counts[2] == approx(n / 3, rel=0.1)

    #
    # Whens
    #

    # I sample using non-uniform probabilities
    counts, key = sample([0.1, 0.3, 0.6], key)

    #
    # Thens
    #

    # Counts should match probabilities
    assert counts[0] == approx(n / 10, rel=0.1)
    assert counts[1] == approx(3 * n / 10, rel=0.1)
    assert counts[2] == approx(6 * n / 10, rel=0.1)
