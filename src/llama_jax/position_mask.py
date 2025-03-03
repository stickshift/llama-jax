from jax import Array
from jax import numpy as jnp

__all__ = [
    "create",
]


def create(seed: Array, *, max_tokens: int) -> Array:
    """Create position mask from seed."""
    # Sanity check seed should be (bs, n)
    assert seed.ndim == 2

    mask = jnp.concat(
        [
            seed,
            jnp.ones((seed.shape[0], max_tokens - seed.shape[1])),
        ],
        axis=-1,
    )

    mask = mask.astype(int)

    return mask
