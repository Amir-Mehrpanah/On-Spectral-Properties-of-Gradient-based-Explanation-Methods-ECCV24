from typing import Dict, List, Callable, Tuple
import jax


def make_explanation_stream(
    *,
    stream_processes: List[Callable],
    stream_head: Dict[str, jax.Array] = {},
):
    """
    args:
        stream_processes: a list of functions that have a standardized signature (key, stream)
        stream_head: the head of the stream
        which take a stream and key and put a mask in the stream.
    returns:
        A function that takes a key and returns a stream with all processes applied.
    the stream is a dictonary of stateless objects shared between all processes and
    accross different runs of the returned explanation stream for better performance.

    all functions in stream_processes should be stateless only depend on the stream and key.
    """
    stream = stream_head.copy()

    def explanation_stream(
        *,
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        for process in stream_processes:
            process(stream=stream, key=key)
        return stream

    return explanation_stream


def make_resize_mask(
    *,
    name: str,
    source_name: str,
    shape: Tuple,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> None:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        shape: shape of the mask
        method: method of interpolation
    returns:
        An inplace function that takes a stream and key and resizes the source mask
        and puts the resized mask in the stream.
    """
    assert (
        shape[0] == 1 and len(shape) == 4
    ), "shape should be a 4D array of shape (1,H,W,C)"

    def resize_mask(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update(
            {
                name: jax.image.resize(
                    stream[source_name],
                    shape=shape,
                    method=method,
                )
            }
        )

    return resize_mask


def make_multiply_masks(
    *,
    name: str,
    source_name: str,
    target_name: str,
) -> None:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
    returns:
        An inplace function that takes a stream and key and multiplies the source mask
        and the target mask and puts the multiplied mask in the stream.
    """

    def multiply_masks(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: stream[source_name] * stream[target_name]})

    return multiply_masks


def make_add_masks(
    *,
    name: str,
    source_name: str,
    target_name: str,
) -> None:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
    returns:
        An inplace function that takes a stream and key and adds the source mask
        and the target mask and puts the added masks in the stream.
    """

    def add_masks(
        *,
        stream: Dict[str, jax.Array],
        key: jax.random.KeyArray,
    ) -> Dict[str, jax.Array]:
        stream.update({name: stream[source_name] + stream[target_name]})

    return add_masks


def make_convex_combination_mask(
    *,
    name: str,
    source_name: str,
    target_name: str,
    alpha_name: str,
) -> None:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
        alpha_name: name of the alpha mask
    returns:
        An inplace function that takes a stream and key and interpolates the source mask
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

    return convex_combination_mask


def make_linear_combination_mask(
    *,
    name: str,
    source_name: str,
    target_name: str,
    alpha_source_name: str,
    alpha_target_name: str,
) -> None:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
        alpha_source_name: name of the source alpha mask
        alpha_target_name: name of the target alpha mask
    returns:
        An inplace function that takes a stream and key and computes the linear combination of
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

    return linear_combination_mask
