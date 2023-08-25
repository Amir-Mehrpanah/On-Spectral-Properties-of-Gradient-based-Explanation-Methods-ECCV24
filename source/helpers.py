import copy
from functools import partial
import json
import os
from typing import Dict, Tuple
from collections import namedtuple
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import flaxmodels as fm
import jax.numpy as jnp
from functools import partial, update_wrapper

from source import operations
from source.configs import DefaultArgs
from source.explanation_methods import noise_interpolation


class AbstractFunction:
    def __init__(self, func) -> None:
        self.func = func
        self.params = {}
        update_wrapper(self, func)

    def __call__(self, **kwargs):
        self.params.update(kwargs)
        return self

    def concretize(self):
        return partial(self.func, **self.params)


class StreamNames:
    batch_index = 10  # does it affect performance if we use a string instead of an int?
    vanilla_grad_mask = 11
    results_at_projection = 12
    log_probs = 13


class Statistics:
    none = 0
    meanx = 1
    meanx2 = 2
    abs_delta = 3


Stream = namedtuple("Stream", ["name", "statistic"])


def gather_stats(
    seed,
    abstract_sampling_process: operations.AbstractFunction,
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
    abstract_sampling_process: operations.AbstractFunction,
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


@operations.AbstractFunction
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
    jax.debug.print("sampled_batch {}", batch_index)
    return stats


@operations.AbstractFunction
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


@operations.AbstractFunction
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

    for key in stream_keys:
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


def noise_interpolation_parser(base_parser):
    base_parser.add_argument(
        "--alpha",
        type=float,
        required=True,
    )
    args = base_parser.parse_args()

    kwargs = {
        "alpha": args.alpha,
        "forward": args.forward,
        "num_classes": args.num_classes,
        "input_shape": args.input_shape,
        "image": args.image,
        "label": args.label,
    }

    # just for printing
    pretty_kwargs = copy.deepcopy(kwargs)
    pretty_kwargs["method"] = "noise_interpolation"
    pretty_kwargs["max_batches"] = args.max_batches
    pretty_kwargs["batch_size"] = args.batch_size
    pretty_kwargs["seed"] = args.seed

    del pretty_kwargs["forward"]
    del pretty_kwargs["image"]
    del pretty_kwargs["label"]

    print("noise_interpolation kwargs:")
    print(json.dumps(pretty_kwargs, sort_keys=True))

    return args, kwargs


def base_parser(parser, default_args: DefaultArgs):
    parser.add_argument(
        "--forward",
        type=str,
        default=default_args.forward,
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=default_args.max_batches,
    )
    parser.add_argument(
        "--method",
        type=str,
        default=default_args.method,
        choices=default_args.methods,
    )
    parser.add_argument(
        "--min_change",
        type=float,
        default=default_args.min_change,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_args.seed,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args.batch_size,
    )
    parser.add_argument(
        "--save_raw_data_dir",
        type=str,
        default=default_args.save_raw_data_dir,
    )
    parser.add_argument(
        "--relative_save_path",
        type=str,
        required=True,
        help="relative path to save raw data will be appended to save_raw_data_dir",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=default_args.num_classes,
    )
    parser.add_argument(
        "--input_shape",
        nargs=4,
        type=int,
        default=default_args.input_shape,
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=default_args.image_index,
    )
    parser.add_argument(
        "--monitored_statistic",
        type=str,
        default=default_args.monitored_statistic,
    )
    parser.add_argument(
        "--monitored_stream",
        type=str,
        default=default_args.monitored_stream,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_args.dataset,
    )
    parser.add_argument(
        "--dataset_skip_index",
        type=int,
        default=default_args.dataset_skip_index,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_args.dataset_dir,
    )

    args = parser.parse_args()
    args = _base_process_args(parser, args)

    return args


def _base_process_args(parser, args):
    os.makedirs(args.save_raw_data_dir, exist_ok=True)
    os.makedirs(args.save_raw_data_dir, exist_ok=True)

    args.input_shape = tuple(args.input_shape)

    inplace_update_query_dataset(args)
    inplace_update_init_forward(args)
    inplace_update_init_method(args, parser)

    if args.monitored_statistic == "meanx2":
        monitored_statistic = Statistics.meanx2
    else:
        raise NotImplementedError("other stats must be implemented")

    if args.monitored_stream == "vanilla_grad_mask":
        monitored_stream = StreamNames.vanilla_grad_mask
    else:
        raise NotImplementedError("other streams must be implemented")

    args.monitored_statistic_key = Stream(
        monitored_stream,
        Statistics.abs_delta,
    )
    args.monitored_statistic_source_key = Stream(
        monitored_stream,
        monitored_statistic,
    )
    args.batch_index_key = Stream(
        StreamNames.batch_index,
        Statistics.none,
    )
    args.stats = {
        Stream(
            StreamNames.vanilla_grad_mask,
            Statistics.meanx,
        ): jnp.zeros(shape=args.input_shape),
        args.monitored_statistic_source_key: jnp.zeros(shape=args.input_shape),
        Stream(
            StreamNames.results_at_projection,
            Statistics.meanx,
        ): jnp.zeros(shape=()),
        Stream(
            StreamNames.results_at_projection,
            Statistics.meanx2,
        ): jnp.zeros(shape=()),
        Stream(
            StreamNames.log_probs,
            Statistics.meanx,
        ): jnp.zeros(shape=(1, args.num_classes)),
        Stream(
            StreamNames.log_probs,
            Statistics.meanx2,
        ): jnp.zeros(shape=(1, args.num_classes)),
        args.monitored_statistic_key: jnp.inf,
        args.batch_index_key: 0,
    }

    return args


def inplace_update_init_method(args, base_parser):
    method, method_parser = str_to_method_and_parser_switch[args.method]
    args, kwargs = method_parser(base_parser)
    args.abstract_process = method(**kwargs)
    return args


def init_resnet50_forward(args):
    resnet50 = fm.ResNet50(
        output="log_softmax",
        pretrained="imagenet",
    )
    params = resnet50.init(
        jax.random.PRNGKey(0),
        jnp.empty(args.input_shape, dtype=jnp.float32),
    )
    resnet50_forward = partial(
        resnet50.apply,
        params,
        train=False,
    )

    args.forward = resnet50_forward


def preprocess(x, img_size):
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=img_size,
        width=img_size,
    )(x)
    x = jnp.array(x)
    x = jnp.expand_dims(x, axis=0) / 255.0
    return x


def query_imagenet(args):
    dataset = tfds.folder_dataset.ImageFolder(root_dir=args.dataset_dir)
    dataset = dataset.as_dataset(split="val", shuffle_files=False)
    dataset = dataset.skip(args.dataset_skip_index)
    base_stream = next(dataset.take(1).as_numpy_iterator())

    image_height = args.input_shape[1]  # (N, H, W, C)
    base_stream["image"] = preprocess(base_stream["image"], image_height)

    args.image = base_stream["image"]
    args.label = base_stream["label"]


def inplace_update_query_dataset(args):
    init_dataset_func = str_to_dataset_query_func_switch[args.dataset]
    return init_dataset_func(args)


def inplace_update_init_forward(args):
    init_forward_func = str_to_init_forward_func_switch[args.forward]
    return init_forward_func(args)


# to avoid if-else statements
str_to_init_forward_func_switch = {
    "resnet50": init_resnet50_forward,
}
str_to_dataset_query_func_switch = {
    "imagenet": query_imagenet,
}
str_to_method_and_parser_switch = {
    "noise_interpolation": (noise_interpolation, noise_interpolation_parser),
}
