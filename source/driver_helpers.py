import copy

import json
import os
from typing import Dict, Tuple
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import flaxmodels as fm
import jax.numpy as jnp

from source.configs import DefaultArgs
from source.explanation_methods import noise_interpolation
from source.utils import (
    Stream,
    StreamNames,
    Statistics,
)

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
