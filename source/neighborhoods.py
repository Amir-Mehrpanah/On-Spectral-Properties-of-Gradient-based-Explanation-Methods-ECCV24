from typing import Callable, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import tensorflow as tf
import chex


def make_deterministic_mask(
    *,
    name: str,
    mask: jnp.ndarray,
) -> None:
    """
    args:
        name: name of the mask
        mask: mask to be used
    returns:
        An inplace function that takes a stream and puts the deterministic mask in the stream.
    """

    assert (
        mask.shape[0] == 1 and mask.ndim == 4
    ), "mask should be a 4D array of shape (1,H,W,C)"

    def deterministic_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: mask})

    return deterministic_mask


def make_uniform_mask(
    *,
    name: str,
    shape: Tuple,
) -> None:
    """
    args:
        name: name of the mask
        shape: shape of the mask
    returns:
        An inplace function that takes a stream and key and samples a mask from the uniform distribution
        depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"

    def uniform_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: jax.random.uniform(key, shape=shape)})

    return uniform_mask


def make_bernoulli_mask(
    *,
    name: str,
    shape: Tuple,
    p: Union[float, jax.Array],
) -> None:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        p: probability of the bernoulli distribution for each element of the mask
    returns:
        An inplace function that takes a stream and key and samples a mask from the bernoulli distribution
        for each element of the mask depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    p = p * jnp.ones(shape=shape) if jnp.isscalar(p) else p
    assert p.shape == shape, "p should have the same shape as the mask"
    assert ((0 <= p) & (p <= 1)).all(), "p should be between 0 and 1"

    def bernoulli_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: jax.random.bernoulli(key, p, shape=shape)})

    return bernoulli_mask


def make_onehot_categorical_mask(
    *,
    name: str,
    shape: Tuple,
    p: Union[float, jax.Array] = None,
) -> None:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        p: probability of the categorical distribution defaults to `None`.
        If `None` the probability distribution is uniform over classes.
        If p is provided, it should be a 4D array of shape (1,H,W,C) and
        equal to the number of classes and should sum to one.
    returns:
        An inplace function that takes a stream and key and samples a mask from the one hot categorical distribution
        depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    spatial_shape = shape[1], shape[2]
    assert not jnp.isscalar(p), (
        "p should not be a scalar because it gives equal probability to all classes."
        " leave it empty to sample from a uniform distribution over classes"
    )
    assert p is not None or (p.sum() == 1.0 and p.shape == spatial_shape), (
        "if p is provided, p should sum to one and p.shape must be"
        "equal to the spatial dimensions of the provided shape"
    )
    p = (
        jax.nn.softmax(jnp.ones(shape=spatial_shape).flatten())
        if p is None
        else p.flatten()
    )
    logits = jnp.log(p)
    static_mask = jnp.zeros(shape=shape)

    def onehot_categorical_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        flat_index = jax.random.categorical(key, logits)
        x_index, y_index = jnp.unravel_index(flat_index, shape=spatial_shape)
        stream.update({name: static_mask.at[0, x_index, y_index, :].set(1.0)})

    return onehot_categorical_mask


def make_normal_mask(
    *,
    name: str,
    shape: Tuple,
) -> None:
    """
    args:
        name: name of the mask
        shape: shape of the mask
    returns:
        An inplace function that takes a stream and key and samples a mask from the normal distribution
        depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"

    def normal_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: jax.random.normal(key, shape=shape)})

    return normal_mask
