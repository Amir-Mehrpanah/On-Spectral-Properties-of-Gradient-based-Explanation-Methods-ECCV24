from typing import Dict, Tuple, Union
import jax
import numpy as np
import jax.numpy as jnp

from source.operations import AbstractProcess


@AbstractProcess
def deterministic_mask(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    mask: jnp.ndarray,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        mask: mask to be used
        key: key to be used for sampling
    returns:
        the resulting stream

    An inplace function that takes a stream and puts the deterministic mask in the stream.
    """

    assert (
        mask.shape[0] == 1 and mask.ndim == 4
    ), "mask should be a 4D array of shape (1,H,W,C)"

    stream.update({name: mask})
    return stream


@AbstractProcess
def uniform_mask(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    shape: Tuple,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        key: key to be used for sampling
    returns:
        the resulting stream

    An inplace function that takes a stream and key and samples a mask from the uniform distribution
    depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"

    stream.update({name: jax.random.uniform(key, shape=shape)})
    return stream


@AbstractProcess
def bernoulli_mask(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    shape: Tuple,
    p: Union[float, np.ndarray],
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        p: probability of the bernoulli distribution for each element of the mask
        key: key to be used for sampling
    returns:
        the resulting stream

    An inplace function that takes a stream and key and samples a mask from the bernoulli distribution
    for each element of the mask depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    p = p * np.ones(shape=shape) if np.isscalar(p) else p
    assert p.shape == shape, "p should have the same shape as the mask"
    assert ((0 <= p) & (p <= 1)).all(), "p should be between 0 and 1"

    stream.update({name: jax.random.bernoulli(key, p, shape=shape)})
    return stream


@AbstractProcess
def onehot_categorical_mask(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    shape: Tuple,
    p: Union[float, np.ndarray] = None,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        p: probability of the categorical distribution defaults to `None`.
        If `None` the probability distribution is uniform over classes.
        If p is provided, it should be a 4D array of shape (1,H,W,C) and
        equal to the number of classes and should sum to one.
        key: key to be used for sampling
    returns:
        the resulting stream

    An inplace function that takes a stream and key and samples a mask from the one hot categorical distribution
    depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    spatial_shape = shape[1], shape[2]
    assert not np.isscalar(p), (
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

    flat_index = jax.random.categorical(key, logits)
    x_index, y_index = jnp.unravel_index(flat_index, shape=spatial_shape)
    stream.update({name: static_mask.at[0, x_index, y_index, :].set(1.0)})
    return stream


@AbstractProcess
def normal_mask(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    shape: Tuple,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        shape: shape of the mask
        key: key to be used for sampling
    returns:
        the resulting stream

    An inplace function that takes a stream and key and samples a mask from the normal distribution
    depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"

    stream.update({name: jax.random.normal(key, shape=shape)})
    return stream
