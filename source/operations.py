from typing import Any, Dict, List, Callable, Tuple
import jax.numpy as jnp
import jax

from source.utils import Stream, StreamNames, Statistics, AbstractFunction


@AbstractFunction
def deterministic_projection(
    *,
    name: str,
    stream=Dict[str, jax.Array],
    projection: jnp.ndarray,
    key: jax.random.KeyArray,
) -> Dict[str, jax.Array]:
    """
    args:
        name: name of the mask
        projection: projection matrix to be used of shape (K, 1) where K is the number of classes
        key: key will be ignored
    returns:
        the resulting stream

    An inplace function that takes a stream and puts the deterministic projection matrix in the stream.
    """

    assert (
        projection.shape[-1] == 1 and projection.ndim == 2
    ), "mask should be a 4D array of shape (K, 1)"

    stream.update({name: projection})
    return stream


def concretize_all(*, abstract_processes):
    """
    args:
        abstract_processes: a list of abstract processes that have specified arguments except key.
    """

    return [process.concretize() for process in abstract_processes]


def bind_all(*, abstract_processes, **kwargs):
    """
    args:
        abstract_processes: a list of abstract processes.
    returns:
        None

    the list of abstract processes will be bound to the kwargs. this function is inplace.
    """
    for process in abstract_processes:
        process(**kwargs)


# jax vmap does not support kwargs this makes our code less elegant
# otherwise we could have used **kwargs in the abstract processes
@AbstractFunction
def sequential_call(key, *, concrete_processes):
    """
    args:
        key: key to be used for sampling
        concrete_processes: a list of concrete processes that have specified arguments except key.
    returns:
        the resulting stream
    """
    for concrete_process in concrete_processes:
        concrete_process(key=key)


@AbstractFunction
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


@AbstractFunction
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


@AbstractFunction
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


def convex_combination_mask(
    *,
    source_mask: str,
    target_mask: str,
    alpha_mask: str,
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
    return (1 - alpha_mask) * source_mask + alpha_mask * target_mask


@AbstractFunction
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


def gather_stats(
    seed,
    abstract_sampling_process: AbstractFunction,
    batch_size,
    max_batches,
    min_change,
    stats: Dict[Stream, jax.Array],
    monitored_statistic_source_key: Stream,
    monitored_statistic_key: Stream,
    batch_index_key,
):
    assert monitored_statistic_key.statistic == Statistics.abs_delta
    assert stats[monitored_statistic_key] == jnp.inf

    (
        loop_initials,
        concrete_stopping_condition,
        concrete_sample_and_update,
    ) = init_loop(
        seed,
        abstract_sampling_process,
        batch_size,
        max_batches,
        min_change,
        stats,
        monitored_statistic_source_key,
        monitored_statistic_key,
        batch_index_key,
    )
    stats = jax.lax.while_loop(
        cond_fun=concrete_stopping_condition,
        body_fun=concrete_sample_and_update,
        init_val=loop_initials,
    )
    return stats


def init_loop(
    seed,
    abstract_sampling_process: AbstractFunction,
    batch_size,
    max_batches,
    min_change,
    stats: Stream,
    monitored_statistic_source_key: Stream,
    monitored_statistic_key: Stream,
    batch_index_key,
):
    # concretize abstract stopping condition
    concrete_stopping_condition = stopping_condition(
        max_batches=max_batches,
        min_change=min_change,
        monitored_statistic_key=monitored_statistic_key,
        batch_index_key=batch_index_key,
    ).concretize()

    # concretize abstract sampling process
    concrete_sampling_process = abstract_sampling_process.concretize()
    vectorized_concrete_sampling_process = jax.vmap(
        concrete_sampling_process,
        in_axes=(0),
    )

    # concretize abstract update stats
    concrete_update_stats = update_stats(
        stream_keys=tuple(stats.keys()),
        monitored_statistic_source_key=monitored_statistic_source_key,
        monitored_statistic_key=monitored_statistic_key,
    ).concretize()

    # concretize abstract sample and update
    concrete_sample_and_update_stats = sample_and_update_stats(
        seed=seed,
        batch_size=batch_size,
        concrete_vectorized_process=vectorized_concrete_sampling_process,
        concrete_update_stats=concrete_update_stats,
        batch_index_key=batch_index_key,
    ).concretize()

    return stats, concrete_stopping_condition, concrete_sample_and_update_stats


@AbstractFunction
def sample_and_update_stats(
    stats,
    *,
    seed,
    batch_size,
    concrete_vectorized_process,
    concrete_update_stats,
    batch_index_key,
):
    stats[batch_index_key] += 1  # lookup
    batch_index = stats[batch_index_key]  # lookup

    key = jax.random.PRNGKey(seed + batch_index)
    batch_keys = jax.random.split(key, num=batch_size)

    sampled_batch = concrete_vectorized_process(batch_keys)
    stats = concrete_update_stats(sampled_batch, stats, batch_index)
    return stats


@AbstractFunction
def stopping_condition(
    stats,
    *,
    max_batches,
    min_change,
    monitored_statistic_key: Stream,
    batch_index_key,
):
    change = stats[monitored_statistic_key]  # lookup
    batch_index = stats[batch_index_key]  # lookup

    value_condition = change > min_change
    iteration_condition = batch_index < max_batches

    return value_condition & iteration_condition


@AbstractFunction
def update_stats(
    sampled_batch: Dict[StreamNames, jax.Array],
    stats: Dict[Stream, jax.Array],
    batch_index: int,
    *,
    stream_keys: Tuple[Stream],
    monitored_statistic_source_key: Stream,
    monitored_statistic_key: Stream,
):
    monitored_statistic_old = stats[monitored_statistic_source_key]  # lookup

    for key in stream_keys:  # optimize key, operation in stream.items():
        if key.statistic == Statistics.meanx:
            stats[key] = (1 / batch_index) * sampled_batch[key.name].mean(axis=0) + (
                (batch_index - 1) / batch_index
            ) * stats[key]
        elif key.statistic == Statistics.meanx2:
            stats[key] = (1 / batch_index) * (sampled_batch[key.name] ** 2).mean(
                axis=0
            ) + ((batch_index - 1) / batch_index) * stats[key]

    monitored_statistic_new = stats[monitored_statistic_source_key]  # lookup

    stats[monitored_statistic_key] = jnp.abs(
        monitored_statistic_new - monitored_statistic_old
    ).max()  # lookup
    return stats
