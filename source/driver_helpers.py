import argparse
import copy
from datetime import datetime

import json
import os
from typing import Dict, Tuple
from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import flaxmodels as fm
import jax.numpy as jnp

from source.configs import DefaultArgs
from source.explanation_methods import noise_interpolation
from source.utils import (
    Switch,
    Stream,
    StreamNames,
    Statistics,
)

methods_switch = Switch()
args_selector_switch = Switch()
inplace_method_parser_switch = Switch()

methods_switch.register(
    "noise_interpolation",
    noise_interpolation.noise_interpolation,
)
inplace_method_parser_switch.register(
    "noise_interpolation",
    noise_interpolation.inplace_noise_interpolation_parser,
)
args_selector_switch.register(
    "noise_interpolation",
    noise_interpolation.noise_interpolation_select_args,
)


def base_parser(parser, default_args: DefaultArgs):
    parser.add_argument(
        "--method",
        type=str,
        default=default_args.method,
        choices=default_args.methods,
    )
    args, _ = parser.parse_known_args()
    inplace_add_method_parser(args.method, parser)

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=default_args.dry_run,
    )
    parser.add_argument(
        "--architecure",
        type=str,
        default=default_args.architecure,
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=default_args.max_batches,
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
        "--save_raw_data_dir",
        type=str,
        default=default_args.save_raw_data_dir,
    )
    parser.add_argument(
        "--save_metadata_dir",
        type=str,
        default=default_args.save_metadata_dir,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_args.dataset_dir,
    )

    args = parser.parse_args()
    args = _process_args(args)

    return args


def _process_args(args):
    os.makedirs(args.save_raw_data_dir, exist_ok=True)
    os.makedirs(args.save_raw_data_dir, exist_ok=True)

    args.input_shape = tuple(args.input_shape)

    inplace_update_query_dataset(args)
    inplace_update_init_forward(args)

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

    inplace_update_method_and_kwargs(args)

    pretty_print_args(args)

    return args


def pretty_print_args(args: argparse.Namespace):
    pretty_kwargs = vars(copy.deepcopy(args))
    pretty_kwargs["method"] = args.method
    pretty_kwargs["max_batches"] = args.max_batches
    pretty_kwargs["batch_size"] = args.batch_size
    pretty_kwargs["seed"] = args.seed

    pretty_kwargs["forward"] = str(args.forward)[:50]
    pretty_kwargs["image"] = str(args.image.shape)
    pretty_kwargs["abstract_process"] = str(args.abstract_process)
    pretty_kwargs["stats"] = f"stats of len {len(args.stats)}"
    pretty_kwargs.update(
        {k: int(v) for k, v in pretty_kwargs.items() if isinstance(v, np.int64)}
    )

    print("experiment args:", json.dumps(pretty_kwargs, indent=4, sort_keys=True))


def inplace_add_method_parser(method, base_parser):
    method_parser = inplace_method_parser_switch[method]
    method_parser(base_parser)


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
    init_forward_func = str_architecure_to_init_forward_func_switch[args.architecure]
    return init_forward_func(args)


def inplace_update_method_and_kwargs(args):
    method = methods_switch[args.method]
    select_kwargs = args_selector_switch[args.method]
    kwargs = select_kwargs(args)
    args.abstract_process = method(**kwargs)


def save_stats(args, stats):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    get_npy_file_path = lambda key: f"{timestamp}.{key}.npy"
    npy_file_paths = []

    args.batch_index = args.stats[args.batch_index_key]
    args.monitored_statistic = args.stats[args.monitored_statistic_key]
    del args.stats[args.batch_index_key]
    del args.stats[args.monitored_statistic_key]
    del args.batch_index_key
    del args.monitored_statistic_source_key
    del args.monitored_statistic_key

    for key, value in stats.items():
        npy_file_path = os.path.join(
            args.save_raw_data_dir, get_npy_file_path(f"{key.name}.{key.statistic}")
        )
        np.save(npy_file_path, value)
        npy_file_paths.append(npy_file_path)
    args.paths = npy_file_paths
    print("saved the raw data to", get_npy_file_path("*"))
    return timestamp, npy_file_paths


def save_metadata(args, name_prefix):
    csv_file_name = f"{name_prefix}.csv"
    csv_file_path = os.path.join(args.save_metadata_dir, csv_file_name)
    args.input_shape = str(args.input_shape)
    del args.stats
    del args.forward
    del args.image
    del args.abstract_process
    args = vars(args)
    dataframe = pd.DataFrame(args)
    dataframe.to_csv(csv_file_path, index=False)
    print("saved the correspoding meta data to", csv_file_path)


# change this to a switch
str_architecure_to_init_forward_func_switch = {
    "resnet50": init_resnet50_forward,
}
str_to_dataset_query_func_switch = {
    "imagenet": query_imagenet,
}
