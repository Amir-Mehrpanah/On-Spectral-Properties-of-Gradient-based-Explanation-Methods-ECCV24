import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())
from tests.assets.test_config import key, in_shape
from source import neighborhoods
from source import operations


def test_stream_sampling():
    base_stream = {}
    base_key = key
    num_samples = 10
    keys = jax.random.split(base_key, num=num_samples)
    neighborhoods.make_deterministic_mask(
        name="alpha_mask",
        mask=0.5 * jnp.ones(shape=(1, 1, 1, 1)),
    )(stream=base_stream, key=keys[0])

    stream_processes = [
        neighborhoods.make_uniform_mask(
            name="uniform_mask",
            shape=(1, 224, 224, 3),
        ),
        neighborhoods.make_bernoulli_mask(
            name="bernoulli_mask",
            shape=(1, 10, 10, 1),
            p=0.5,
        ),
        operations.make_resize_mask(
            name="bernoulli_mask_resized",
            source_name="bernoulli_mask",
            shape=(1, 224, 224, 1),
        ),
        operations.make_convex_combination_mask(
            name="convex_combination_mask",
            source_name="uniform_mask",
            target_name="bernoulli_mask_resized",
            alpha_name="alpha_mask",
        ),
    ]
    explanation_stream = operations.make_explanation_stream(
        stream_head=base_stream,
        stream_processes=stream_processes,
    )

    expected_stream = base_stream.copy()
    for process in stream_processes:
        process(stream=expected_stream, key=keys[0])
    stream_0 = explanation_stream(key=keys[0])

    assert stream_0 is not expected_stream
    assert stream_0 is not base_stream
    assert stream_0.keys() == expected_stream.keys()
    assert (
        stream_0["convex_combination_mask"]
        == expected_stream["convex_combination_mask"]
    ).all()

    vmap_explanation_stream = jax.vmap(explanation_stream, in_axes=0)
    stream_1 = vmap_explanation_stream(key=keys)
    assert stream_1 is not stream_0
    assert stream_1.keys() == expected_stream.keys()
    assert stream_1["convex_combination_mask"].shape[0] == num_samples
    assert (
        stream_1["convex_combination_mask"].shape[1:]
        == expected_stream["convex_combination_mask"].shape
    )
    assert (
        stream_1["convex_combination_mask"][0]
        == expected_stream["convex_combination_mask"]
    ).all()


def test_resize_mask():
    resize = operations.make_resize_mask(
        name="test_mask",
        shape=in_shape,
        source_name="small_mask",
    )
    small_shape = (1, 5, 5, 3)
    small_mask = jax.random.uniform(key, shape=small_shape)
    expected = jax.image.resize(
        small_mask,
        shape=in_shape,
        method=jax.image.ResizeMethod.LINEAR,
    )
    out = {"small_mask": small_mask}
    resize(stream=out, key=key)
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()


def test_convex_combination_mask():
    convex_combination = operations.make_convex_combination_mask(
        name="test_mask",
        source_name="input",
        target_name="target",
        alpha_name="alpha",
    )
    key_1, key_2, key_3 = jax.random.split(key, num=3)
    input = jax.random.uniform(key_1, shape=in_shape)
    target = jax.random.uniform(key_2, shape=in_shape)
    alpha = jax.random.uniform(key_3)
    expected = (1 - alpha) * input + (alpha) * target
    out = {"input": input, "target": target, "alpha": alpha}
    convex_combination(
        stream=out,
        key=key,
    )
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()


def test_linear_combination_mask():
    linear_combination = operations.make_linear_combination_mask(
        name="test_mask",
        source_name="input",
        target_name="target",
        alpha_source_name="alpha_source",
        alpha_target_name="alpha_target",
    )
    key_1, key_2, key_3, key_4 = jax.random.split(key, num=4)
    input = jax.random.uniform(key_1, shape=in_shape)
    target = jax.random.uniform(key_2, shape=in_shape)
    alpha_source = jax.random.uniform(key_3)
    alpha_target = jax.random.uniform(key_4)
    expected = alpha_source * input + alpha_target * target
    out = {
        "input": input,
        "target": target,
        "alpha_source": alpha_source,
        "alpha_target": alpha_target,
    }
    linear_combination(
        stream=out,
        key=key,
    )
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()


def test_vmap():
    dct = {"a": 0.0, "b": 3}
    x = jax.random.split(jax.random.PRNGKey(0), num=10)

    def foo_factory(dct):
        def foo(x):
            dct.update(
                {
                    "c": dct["b"]
                    + jax.random.bernoulli(
                        x,
                    )
                }
            )
            # without return, we can only use dct which remains
            # a batch tracer object after the function call.
            return dct

        return foo

    foo = foo_factory(dct)
    out = jax.vmap(foo, in_axes=0)
    u = out(x)
    assert u["c"].shape == (10,)
