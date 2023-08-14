from functools import partial, update_wrapper
from typing import Any, Dict, List, Callable, Tuple
import jax


class AbstractProcess:
    def __init__(self, func) -> None:
        self.func = func
        self.params = {}
        update_wrapper(self, func)

    def __call__(self, **kwargs):
        self.params.update(kwargs)
        return self

    def concretize(self):
        return partial(self.func, **self.params)


def concretize(*, abstract_processes):
    """
    args:
        abstract_processes: a list of abstract processes that have specified arguments except key.
    """

    return [process.concretize() for process in abstract_processes]


def count_compilations(func):
    def wrapper(*args, **kwargs):
        wrapper.number_of_compilations += 1
        return func(*args, **kwargs)

    wrapper.number_of_compilations = 0

    return wrapper


# jax vmap does not support kwargs this makes our code less elegant
# otherwise we could have used **kwargs in the abstract processes
@AbstractProcess
def sequential_call(key, stream, *, concrete_processes):
    """
    args:
        key: key to be used for sampling
        stream: stream to be used
        concrete_processes: a list of concrete processes that have specified arguments except key.
    returns:
        the resulting stream
    """
    for concrete_process in concrete_processes:
        concrete_process(stream=stream, key=key)
    return stream


@AbstractProcess
def resize_mask(
    *,
    name: str,
    stream: Dict[str, jax.Array],
    source_name: str,
    shape: Tuple,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
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

    stream.update(
        {
            name: jax.image.resize(
                stream[source_name],
                shape=shape,
                method=method,
            )
        }
    )
    return stream


@AbstractProcess
def multiply_masks(
    *,
    name: str,
    stream: Dict[str, jax.Array],
    source_name: str,
    target_name: str,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
    returns:
        An inplace function that takes a stream and key and multiplies the source mask
        and the target mask and puts the multiplied mask in the stream.
    """

    stream.update({name: stream[source_name] * stream[target_name]})
    return stream


@AbstractProcess
def add_masks(
    *,
    name: str,
    stream: Dict[str, jax.Array],
    source_name: str,
    target_name: str,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        source_name: name of the source mask
        target_name: name of the target mask
    returns:
        An inplace function that takes a stream and key and adds the source mask
        and the target mask and puts the added masks in the stream.
    """

    stream.update({name: stream[source_name] + stream[target_name]})
    return stream


@AbstractProcess
def convex_combination_mask(
    *,
    name: str,
    stream: Dict[str, jax.Array],
    source_name: str,
    target_name: str,
    alpha_name: str,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
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
    stream.update(
        {
            name: (1 - stream[alpha_name]) * stream[source_name]
            + stream[alpha_name] * stream[target_name]
        }
    )
    return stream


@AbstractProcess
def linear_combination_mask(
    *,
    name: str,
    stream: Dict[str, jax.Array],
    source_name: str,
    target_name: str,
    alpha_source_name: str,
    alpha_target_name: str,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
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

    stream.update(
        {
            name: stream[alpha_source_name] * stream[source_name]
            + stream[alpha_target_name] * stream[target_name]
        }
    )
    return stream
