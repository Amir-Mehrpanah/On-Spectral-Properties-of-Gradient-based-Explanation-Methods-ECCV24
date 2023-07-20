from typing import Callable, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import tensorflow as tf
import chex


def make_deterministic_mask(
    *,
    name: str,
    mask: jnp.ndarray,
) -> Callable:
    """
    Args:
        name: name of the mask
        mask: mask to be used
    returns:
        A function that takes a stream and puts the deterministic mask in the stream.
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
        return stream

    return deterministic_mask


def get_uniform_mask(
    *,
    name: str,
    shape: Tuple,
) -> Callable:
    """
    Args:
        name: name of the mask
        shape: shape of the mask
    returns:
        A function that takes a stream and key and samples a mask from the uniform distribution
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
        return stream

    return uniform_mask


def get_bernoulli_mask(
    *,
    name: str,
    shape: Tuple,
    p: Union[float, jax.Array],
) -> Callable:
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    p = p if isinstance(p, jax.Array) else p * jnp.ones(size=shape)
    assert all(0 <= p <= 1), "p should be between 0 and 1"

    def bernoulli_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: jax.random.bernoulli(key, p, shape=shape)})
        return stream

    return bernoulli_mask


def get_onehot_categorical_mask(
    *,
    name: str,
    shape: Tuple,
    p: Union[float, jax.Array],
) -> Callable:
    """
    Args:
        name: name of the mask
        shape: shape of the mask
        p: probability of the bernoulli distribution
    returns:
        A function that takes a stream and key and samples a mask from the one hot categorical distribution
        depending on the key provided and puts the sampled mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    H, W = shape[1], shape[2]
    p = p if isinstance(p, jax.Array) else p * jnp.ones(size=(H, W))
    assert p.sum() == 1, "p should sum to 1"
    p = p.flatten()
    static_mask = jnp.zeros(shape=shape)

    def onehot_categorical_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        flat_index = jax.random.categorical(key, p)
        x_index, y_index = jnp.unravel_index(flat_index, shape=(H, W))
        stream.update({name: static_mask.at[0, x_index, y_index, :].set(1.0)})
        return stream

    return onehot_categorical_mask


def get_normal_mask(
    *,
    name: str,
    shape: Tuple,
) -> Callable:
    """
    Args:
        name: name of the mask
        shape: shape of the mask
    returns:
        A function that takes a stream and key and samples a mask from the normal distribution
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
        return stream

    return normal_mask


def get_resize_mask(
    *,
    name: str,
    source_name: str,
    shape: Tuple,
    method: str = "bilinear",
) -> Callable:
    """
    Args:
        name: name of the mask
        source_name: name of the source mask
        shape: shape of the mask
        method: method of interpolation
    returns:
        A function that takes a stream and key and resizes the source mask
        and puts the resized mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"
    H, W = shape[1], shape[2]

    def resize_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update(
            {
                name: tf.image.resize(
                    stream[source_name],
                    size=(H, W),
                    method=method,
                )
            }
        )
        return stream

    return resize_mask


def get_convex_combination_mask(
    *,
    name: str,
    source_name: str,
    target_name: str,
    alpha_name: str,
) -> Callable:
    """
    Args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
        alpha_name: name of the alpha mask
    returns:
        A function that takes a stream and key and interpolates the source mask
        and the target mask with the alpha mask provided and puts the interpolated
        mask in the stream. `output = source*(1-alpha)+target*(alpha)` if alpha is
        zero, the output is the source mask and when alpha is one, the output is
        the target mask. all masks should have the same spatial shape or be scalars.
    """

    def convex_combination_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update(
            {
                name: (1 - stream[alpha_name]) * stream[source_name]
                + stream[alpha_name] * stream[target_name]
            }
        )
        return stream

    return convex_combination_mask


def get_linear_combination_mask(
    *,
    name: str,
    source_name: str,
    target_name: str,
    alpha_source_name: str,
    alpha_target_name: str,
) -> Callable:
    """
    Args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
        alpha_source_name: name of the source alpha mask
        alpha_target_name: name of the target alpha mask
    returns:
        A function that takes a stream and key and computes the linear combination of
        the source mask and the target mask with the alpha mask provided and puts the
        resulting mask in the stream. `output = alpha_source*source_mask+alpha_target*target_mask`.
        all masks should have the same spatial shape or be scalars.
    """

    def linear_combination_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update(
            {
                name: stream[alpha_source_name] * stream[source_name]
                + stream[alpha_target_name] * stream[target_name]
            }
        )
        return stream

    return linear_combination_mask


def get_explanation_stream(
    stream_head: Dict[str, jax.Array],
    *args: Callable,
):
    """
    Args:
        stream_head: the head of the stream
        args: a list of functions that have a standardized signature
        which take a stream and key and put a mask in the stream.
    returns:
        A function that takes a key and returns a stream with the masks in it.
    """

    def explanation_stream(
        *,
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream = stream_head
        for arg in args:
            stream = arg(stream=stream, key=key)
        return stream

    return explanation_stream
