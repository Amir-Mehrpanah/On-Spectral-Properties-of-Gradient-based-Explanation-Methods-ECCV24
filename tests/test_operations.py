import copy
from functools import wraps
import os
import sys
import jax

import jax.numpy as jnp

sys.path.append(os.getcwd())
from tests.assets.test_config import key, in_shape
from source import neighborhoods
from source import operations


def test_partial_call():
    @operations.AbstractProcess
    def func(*, idict, y):
        """
        test docstring
        """
        idict.update({"y": idict["x"] + y})
        return idict

    func(y=2)
    concrete_func = func.concretize()
    idict = {"x": 5}
    result = concrete_func(idict=idict)

    complied_func = jax.jit(concrete_func)
    compiled_result = complied_func(idict=idict)

    assert id(result) == id(idict)
    assert id(result) != id(compiled_result)  # jitted function returns a new dict
    assert result["y"] == 7
    assert compiled_result["y"] == 7


def test_partial_compile():
    @operations.AbstractProcess
    def func(*, x, y, z):
        """
        test docstring
        """
        return x + y + z

    func(x=1, y=2)
    compiled_func = func.concretize()
    assert compiled_func(z=3) == 6


def test_concrete_process_compilation_count():
    @operations.AbstractProcess
    def func(*, x, y, z):
        """
        test docstring
        """
        return x + y + z

    func(x=1)
    concrete_func = func.concretize()
    concrete_func = operations.count_compilations(concrete_func)
    compiled_func = jax.jit(concrete_func, static_argnames=("x", "y"))

    assert compiled_func(y=1, z=3) == 5
    assert concrete_func.number_of_compilations == 1

    assert compiled_func(y=1, z=4) == 6
    assert concrete_func.number_of_compilations == 1

    assert compiled_func(y=2, z=4) == 7  # force recompilation
    assert concrete_func.number_of_compilations == 2


def get_abstract_stream_sampler(base_stream, keys):
    # initialize a static mask in the stream that does not depend on the key
    concrete_process = neighborhoods.deterministic_mask(
        name="alpha_mask",
        mask=0.5 * jnp.ones(shape=(1, 1, 1, 1)),
        stream=base_stream,
        key=keys[0],
    ).concretize()
    # put the static mask in the stream
    concrete_process()

    # initialize other masks that depend on the key with tailored inputs
    base_abstract_processes = [
        neighborhoods.uniform_mask(
            name="uniform_mask",
            shape=(1, 224, 224, 3),
        ),
        neighborhoods.bernoulli_mask(
            name="bernoulli_mask",
            shape=(1, 10, 10, 1),
            p=0.5,
        ),
        operations.resize_mask(
            name="bernoulli_mask_resized",
            source_name="bernoulli_mask",
            shape=(1, 224, 224, 1),
        ),
        operations.convex_combination_mask(
            name="convex_combination_mask",
            source_name="uniform_mask",
            target_name="bernoulli_mask_resized",
            alpha_name="alpha_mask",
        ),
    ]
    return base_abstract_processes


def get_concrete_stream_sampler(base_stream, keys):
    base_abstract_processes = get_abstract_stream_sampler(base_stream, keys)

    concrete_processes = operations.concretize(
        abstract_processes=base_abstract_processes
    )
    # create a concrete sequential process
    concrete_sequential_process = operations.sequential_call(
        concrete_processes=concrete_processes
    ).concretize()
    return concrete_sequential_process


def test_stream_sampling():
    base_stream = {}
    num_samples = 10
    keys = jax.random.split(key, num=num_samples)

    concrete_sequential_process = get_concrete_stream_sampler(base_stream, keys)

    # compute the expected stream
    expected_stream = copy.deepcopy(base_stream)
    concrete_sequential_process(keys[0], expected_stream)

    # vmap the concrete sequential process
    vmap_concrete_sequential_process = jax.vmap(
        concrete_sequential_process, in_axes=(0, None)
    )

    # count the number of compilations
    vmap_concrete_sequential_process = operations.count_compilations(
        vmap_concrete_sequential_process
    )
    # compile the concrete sequential process and call it
    compiled_concrete_sequential_process = jax.jit(vmap_concrete_sequential_process)
    result_stream = compiled_concrete_sequential_process(keys, base_stream)

    assert vmap_concrete_sequential_process.number_of_compilations == 1
    assert base_stream is not result_stream
    assert result_stream.keys() == expected_stream.keys()
    assert (
        result_stream["convex_combination_mask"][0]
        != result_stream["convex_combination_mask"][1]
    ).all()
    assert result_stream["convex_combination_mask"].shape[0] == num_samples
    assert (
        result_stream["convex_combination_mask"].shape[1:]
        == expected_stream["convex_combination_mask"].shape
    )
    assert (
        result_stream["convex_combination_mask"][0]
        == expected_stream["convex_combination_mask"]
    ).all()


def test_grad_stream_sampling():
    base_stream = {}
    num_samples = 10
    keys = jax.random.split(key, num=num_samples)

    concrete_sequential_process = get_abstract_stream_sampler(base_stream, keys)

    # compute the expected stream
    expected_stream = copy.deepcopy(base_stream)
    concrete_sequential_process(keys[0], expected_stream)

    # vmap the concrete sequential process
    vmap_concrete_sequential_process = jax.vmap(
        concrete_sequential_process, in_axes=(0, None)
    )

    # count the number of compilations
    vmap_concrete_sequential_process = operations.count_compilations(
        vmap_concrete_sequential_process
    )
    # compile the concrete sequential process and call it
    compiled_concrete_sequential_process = jax.jit(vmap_concrete_sequential_process)
    result_stream = compiled_concrete_sequential_process(keys, base_stream)

    assert vmap_concrete_sequential_process.number_of_compilations == 1
    assert base_stream is not result_stream
    assert result_stream.keys() == expected_stream.keys()
    assert (
        result_stream["convex_combination_mask"][0]
        != result_stream["convex_combination_mask"][1]
    ).all()
    assert result_stream["convex_combination_mask"].shape[0] == num_samples
    assert (
        result_stream["convex_combination_mask"].shape[1:]
        == expected_stream["convex_combination_mask"].shape
    )
    assert (
        result_stream["convex_combination_mask"][0]
        == expected_stream["convex_combination_mask"]
    ).all()


def test_resize_mask():
    resize = operations.resize_mask(
        name="test_mask",
        shape=in_shape,
        source_name="small_mask",
    ).concretize()
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
    convex_combination = operations.convex_combination_mask(
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
    )
    concrete_convex_combination = convex_combination.concretize()
    out = concrete_convex_combination(key=key)
    assert out["test_mask"].shape == in_shape
    assert (out["test_mask"] == expected).all()


def test_linear_combination_mask():
    linear_combination = operations.linear_combination_mask(
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
    )
    concrete_linear_combination = linear_combination.concretize()
    out = concrete_linear_combination(key=key)

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
