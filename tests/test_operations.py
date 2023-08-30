import copy
import os
import sys
import jax

import jax.numpy as jnp
import numpy as np

sys.path.append(os.getcwd())
from tests.assets.test_config import key, in_shape
from source import operations
from source.utils import AbstractFunction


def test_partial_call():
    @AbstractFunction
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


def test_resize_mask():
    small_shape = (1, 5, 5, 3)
    small_mask = jax.random.uniform(key, shape=small_shape)
    expected = jax.image.resize(
        small_mask,
        shape=in_shape,
        method=jax.image.ResizeMethod.LINEAR,
    )
    resized_mask = operations.resize_mask(
        source_mask=small_mask,
        shape=in_shape,
    )
    assert resized_mask.shape == in_shape
    np.testing.assert_allclose(resized_mask, expected, rtol=1e-6)


def test_convex_combination_mask():
    key_1, key_2, key_3 = jax.random.split(key, num=3)
    input = jax.random.uniform(key_1, shape=in_shape)
    target = jax.random.uniform(key_2, shape=in_shape)
    alpha = jax.random.uniform(key_3)
    expected = (1 - alpha) * input + (alpha) * target

    convex_combination = operations.convex_combination_mask(
        source_mask=input,
        target_mask=target,
        alpha_mask=alpha,
    )
    assert convex_combination.shape == in_shape
    np.testing.assert_allclose(convex_combination, expected, rtol=1e-6)


def test_linear_combination_mask():
    key_1, key_2, key_3, key_4 = jax.random.split(key, num=4)
    input = jax.random.uniform(key_1, shape=in_shape)
    target = jax.random.uniform(key_2, shape=in_shape)
    alpha_source = jax.random.uniform(key_3)
    alpha_target = jax.random.uniform(key_4)
    expected = alpha_source * input + alpha_target * target

    linear_combination = operations.linear_combination_mask(
        source_mask=input,
        target_mask=target,
        alpha_source_mask=alpha_source,
        alpha_target_mask=alpha_target,
    )

    assert linear_combination.shape == in_shape
    np.testing.assert_allclose(linear_combination, expected, rtol=1e-6)
