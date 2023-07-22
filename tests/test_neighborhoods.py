import jax
import jax.numpy as jnp
import os
import sys

sys.path.append(os.getcwd())
from source import neighborhoods
from tests.assets.test_config import key, in_shape


def test_deterministic_mask():
    sample_mask = jnp.zeros(in_shape)
    mask = neighborhoods.make_deterministic_mask(
        name="test_mask",
        mask=sample_mask,
    )
    out = {}
    mask(stream=out, key=key)
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == sample_mask).all()


def test_uniform_mask():
    mask = neighborhoods.make_uniform_mask(
        name="test_mask",
        shape=in_shape,
    )
    expected = jax.random.uniform(key, shape=in_shape)
    out = {}
    mask(stream=out, key=key)
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()


def test_bernoulli_mask():
    ps = [0.0, 0.5, 1.0]
    ps.append(jnp.ones(shape=in_shape))

    for p in ps:
        mask = neighborhoods.make_bernoulli_mask(
            name="test_mask",
            shape=in_shape,
            p=p,
        )
        expected = jax.random.bernoulli(key, p=p, shape=in_shape)
        out = {}
        mask(stream=out, key=key)
        assert out["test_mask"].shape == in_shape
        assert (out["test_mask"] == expected).all()
        p = jnp.array(p) if jnp.isscalar(p) else p
        assert (out["test_mask"] == 0).all() or (p > 0.0).all()  # p == 0.0 -> all zeros
        assert (out["test_mask"] == 1).all() or (p < 1.0).all()  # p == 1.0 -> all ones


def test_onehot_categorical_mask():
    H, W = in_shape[1], in_shape[2]
    ps = [
        jax.nn.softmax(jnp.ones(shape=(H, W))),
        jnp.zeros(shape=(H, W)).at[0, 0].set(1.0),
    ]
    for p in ps:
        mask = neighborhoods.make_onehot_categorical_mask(
            name="test_mask",
            shape=in_shape,
            p=p,
        )
        flat_index = jax.random.categorical(key, jnp.log(p).flatten())
        x_index, y_index = jnp.unravel_index(flat_index, shape=(H, W))
        expected = jnp.zeros(shape=in_shape).at[0, x_index, y_index, :].set(1.0)
        out = {}
        mask(stream=out, key=key)
        assert out["test_mask"].shape == in_shape
        assert (out["test_mask"] == expected).all()
        assert p.max() < 1.0 or (
            out["test_mask"].argmax() == p.argmax()
        )  # p[i,j] == 1.0 -> out[i,j,:] == 1.0


def test_normal_mask():
    mask = neighborhoods.make_normal_mask(
        name="test_mask",
        shape=in_shape,
    )
    expected = jax.random.normal(key, shape=in_shape)
    out = {}
    mask(stream=out, key=key)
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()
