from typing import Any, Dict, List, Callable, Tuple
import jax.numpy as jnp
import jax

from source.utils import Stream, StreamNames, Statistics, AbstractFunction


def static_projection(*, num_classes, index):
    return (
        jnp.zeros(
            shape=(num_classes, 1),
            dtype=jnp.float32,
        )
        .at[index, 0]
        .set(1.0)
    )


def prediction_projection(*, forward, image, k):
    log_probs = forward(image)
    kth_max = jnp.argpartition(log_probs.squeeze(), -k)[-k]
    return static_projection(num_classes=log_probs.shape[1], index=kth_max)


def resize_mask(
    *,
    source_mask: str,
    shape: Tuple,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> jax.Array:
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

    return jax.image.resize(
        source_mask,
        shape=shape,
        method=method,
    )


def convex_combination_mask(
    *,
    source_mask: str,
    target_mask: str,
    alpha_mask: str,
) -> jax.Array:
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


def linear_combination_mask(
    *,
    source_mask: str,
    target_mask: str,
    alpha_source_mask: str,
    alpha_target_mask: str,
) -> jax.Array:
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

    return alpha_source_mask * source_mask + alpha_target_mask * target_mask


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
