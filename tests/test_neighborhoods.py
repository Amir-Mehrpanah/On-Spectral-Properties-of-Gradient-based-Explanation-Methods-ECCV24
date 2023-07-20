import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.getcwd())
from source import neighborhoods


def test_deterministic_mask():
    key = jax.random.PRNGKey(0)
    mask = jnp.zeros((10, 10))
    mask = neighborhoods.deterministic_mask(key, name="test_dmask", mask=mask)
    assert mask.shape == (10, 10)
    assert mask.sum() == 50
    assert mask.sum() == mask.sum()
