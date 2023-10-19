import argparse
import copy
from datetime import datetime
import json
import os
import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())
from source.configs import DefaultArgs
from source.data_manager import query_imagenet
from source.explanation_methods.noise_interpolation import NoiseInterpolation
from source.model_manager import init_resnet50_forward
from source.utils import (
    Switch,
    Stream,
    StreamNames,
    Statistics,
)


def json_semicolon_loads(string):
    return json.loads(string.replace(";", ","))


methods_switch = Switch()
dataset_query_func_switch = Switch()
init_architecture_forward_switch = Switch()


methods_switch.register(
    "noise_interpolation",
    NoiseInterpolation(),
)
dataset_query_func_switch.register(
    "imagenet",
    query_imagenet,
)
init_architecture_forward_switch.register(
    "resnet50",
    init_resnet50_forward,
)


def base_parser(parser, default_args: DefaultArgs):
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=default_args.methods,
    )
    args, _ = parser.parse_known_args()
    print(_)
    methods_switch[args.method].inplace_add_args(parser)

    add_base_args(parser, default_args)

    args = parser.parse_args()
    args = _process_args(args)

    return args


def add_base_args(parser, default_args):
    parser.add_argument(
        "--no_demo",
        action="store_false",
        dest="write_demo",
        default=default_args.write_demo,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--disable_jit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--assert_device",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=default_args.architectures,
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
        nargs="+",
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
    parser.add_argument(
        "--output_layer",
        type=str,
        default=default_args.output_layer,
        choices=default_args.output_layers,
    )
    parser.add_argument(
        "--stats_log_level",
        type=int,
        default=default_args.stats_log_level,
    )
    parser.add_argument(
        "--args_state",
        type=json_semicolon_loads,
        default=default_args.args_state,
    )
    parser.add_argument(
        "--args_pattern",
        type=json_semicolon_loads,
        default=default_args.args_pattern,
    )
    parser.add_argument(
        "--gather_stats",
        action="store_true",
        default=default_args.gather_stats,
    )


def _process_args(args):
    if args.assert_device:
        assert jax.device_count() > 0, "jax devices are not available"

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)

    if args.dry_run:
        pass
        # jax.config.update("jax_log_compiles", True)
        # jax.config.update('jax_platform_name', 'cpu')

    os.makedirs(args.save_raw_data_dir, exist_ok=True)
    os.makedirs(args.save_metadata_dir, exist_ok=True)

    dataset_query_func_switch[args.dataset](args)
    init_architecture_forward_switch[args.architecture](args)

    if args.monitored_statistic == "meanx2":
        monitored_statistic = Statistics.meanx2
    else:
        raise NotImplementedError("other stats are not implemented")

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
        args.monitored_statistic_source_key: jnp.zeros(shape=args.input_shape),
        Stream(
            StreamNames.results_at_projection,
            Statistics.meanx,
        ): jnp.zeros(shape=()),
        Stream(
            StreamNames.log_probs,
            Statistics.meanx,
        ): jnp.zeros(shape=(1, args.num_classes)),
        args.monitored_statistic_key: jnp.inf,
        args.batch_index_key: 0,
    }

    if args.stats_log_level >= 1:
        if args.stats_log_level >= 2:
            args.stats[
                Stream(
                    StreamNames.log_probs,
                    Statistics.meanx2,
                )
            ] = jnp.zeros(shape=(1, args.num_classes))
            args.stats[
                Stream(
                    StreamNames.results_at_projection,
                    Statistics.meanx2,
                )
            ] = jnp.zeros(shape=())
        args.stats[
            Stream(
                StreamNames.vanilla_grad_mask,
                Statistics.meanx,
            )
        ] = jnp.zeros(shape=args.input_shape)

    inplace_update_method_and_kwargs(args)
    pretty_print_args(args)

    return args


def sampling_demo(args, stats):
    if args.write_demo:
        print("sampling demo")
        methods_switch[args.method].inplace_demo(args, stats)


def pretty_print_args(args: argparse.Namespace):
    pretty_kwargs = copy.deepcopy(args.meta_kwargs)
    inplace_propagate = lambda k, v: [item.update({k: v}) for item in pretty_kwargs]
    inplace_propagate("method", args.method)
    inplace_propagate("max_batches", args.max_batches)
    inplace_propagate("batch_size", args.batch_size)
    inplace_propagate("seed", args.seed)
    temp_stats = f"stats of len {len(args.stats)}"
    inplace_propagate("stats", temp_stats)
    temp_labels = [int(items["label"]) for items in pretty_kwargs]
    inplace_propagate("label", temp_labels)
    methods_switch[args.method].inplace_make_pretty(pretty_kwargs)
    print("experiment args:", json.dumps(pretty_kwargs, indent=4, sort_keys=True))


def inplace_update_method_and_kwargs(args):
    method_cls = methods_switch[args.method]
    method_cls.inplace_process_args(args)


def save_stats(args, stats, sindex):
    path_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    get_npy_file_path = lambda key: os.path.join(
        args.save_raw_data_dir, f"{path_prefix}.{key}.npy"
    )

    # temporary metadata
    npy_file_paths = []
    stream_name = []
    stream_statistic = []
    metadata = {}

    # update metadata
    metadata["batch_index"] = stats[args.batch_index_key]
    metadata["monitored_statistic_change"] = float(stats[args.monitored_statistic_key])

    metadata["path_prefix"] = path_prefix
    csv_file_name = f"{metadata['path_prefix']}.csv"
    metadata_file_path = os.path.join(args.save_metadata_dir, csv_file_name)
    metadata["metadata_file_path"] = metadata_file_path
    metadata["method"] = args.method
    metadata["architecture"] = args.architecture
    metadata["output_layer"] = args.output_layer
    metadata["dataset"] = args.dataset
    metadata["time_to_compute"] = args.time_to_compute
    metakwargs = args.meta_kwargs[sindex]
    metakwargs = {k: w for k, w in metakwargs.items() if w != None}
    metadata.update(metakwargs)
    metadata["label"] = int(metadata["label"])
    metadata["projection_index"] = int(metadata["projection_index"])
    metadata["input_shape"] = str(metadata["input_shape"])

    del stats[args.batch_index_key]
    del stats[args.monitored_statistic_key]

    # save stats
    for key, value in stats.items():
        npy_file_path = get_npy_file_path(f"{key.name}.{key.statistic}")
        np.save(npy_file_path, value.squeeze())

        # update metadata
        npy_file_paths.append(npy_file_path)
        stream_name.append(key.name)
        stream_statistic.append(key.statistic)

    metadata["data_path"] = npy_file_paths
    metadata["stream_name"] = stream_name
    metadata["stream_statistic"] = stream_statistic

    print("saved the raw data to", get_npy_file_path("*"))

    return metadata


def save_metadata(metadata):
    metadata_file_path = metadata["metadata_file_path"]
    # convert metadata from dict to dataframe and save
    dataframe = pd.DataFrame(metadata)
    dataframe.to_csv(metadata_file_path, index=False)
    print("saved the correspoding meta data to", metadata_file_path)
