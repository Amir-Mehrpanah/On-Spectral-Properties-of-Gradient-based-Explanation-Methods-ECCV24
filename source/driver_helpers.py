import argparse
from datetime import datetime
import json
import os
import sys
import logging
from typing import List
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import tensorflow as tf

sys.path.append(os.getcwd())
from source.configs import DefaultArgs
from source.data_manager import query_imagenet
from source.explanation_methods.noise_interpolation import NoiseInterpolation
from source.model_manager import init_resnet50_forward
from source.consistency_measures import (
    _measure_consistency_cosine_distance,
    _measure_consistency_DSSIM,
)
from source.utils import (
    Action,
    ConsistencyMeasures,
    Switch,
    Stream,
    StreamNames,
    Statistics,
)

logger = logging.getLogger(__name__)


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
    args = _parse_general_args(parser, default_args)

    if args.action == Action.gather_stats:
        action_args, write_demo = _parse_gather_stats_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            write_demo=write_demo,
            save_raw_data_dir=args.save_raw_data_dir,
            save_metadata_dir=args.save_metadata_dir,
        )
    elif args.action == Action.compute_consistency:
        action_args = _parse_measure_consistency_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    elif args.action == Action.merge_stats:
        action_args = argparse.Namespace()
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    else:
        raise NotImplementedError("other actions are not implemented")

    return driver_args, action_args


def _parse_measure_consistency_args(parser, default_args):
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args.batch_size,
    )
    parser.add_argument(
        "--pivot_index",
        nargs="+",
        type=str,
        default=default_args.pivot_index,
    )
    parser.add_argument(
        "--pivot_column",
        type=str,
        default=default_args.pivot_column,
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=default_args.prefetch_factor,
    )
    parser.add_argument(
        "--downsampling_factor",
        type=int,
        default=default_args.downsampling_factor,
    )
    parser.add_argument(
        "--consistency_measure",
        type=str,
        required=True,
        choices=default_args.consistency_measures,
    )
    args, _ = parser.parse_known_args()
    data_loader = _make_loader(
        args.save_metadata_dir,
        args.pivot_index,
        args.batch_size,
        args.consistency_measure,
        args.pivot_column,
        prefetch_factor=args.prefetch_factor,
    )
    consistency_measure_func = get_consistency_measure(args)

    return argparse.Namespace(
        data_loader=data_loader,
        pivot_column=args.pivot_column,
        consistency_measure=consistency_measure_func,
        consistency_measure_name=args.consistency_measure,
    )


def get_consistency_measure(args):
    if args.consistency_measure == ConsistencyMeasures.cosine_distance:
        consistency_measure_func = _measure_consistency_cosine_distance(
            downsampling_factor=args.downsampling_factor
        )
    elif args.consistency_measure == ConsistencyMeasures.dssim:
        consistency_measure_func = _measure_consistency_DSSIM(
            downsampling_factor=args.downsampling_factor
        )
    else:
        raise NotImplementedError("other consistency measures are not implemented")
    consistency_measure_func = consistency_measure_func.concretize()
    return consistency_measure_func


def _parse_gather_stats_args(parser, default_args):
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=default_args.methods,
    )
    args, _ = parser.parse_known_args()

    methods_switch[args.method].inplace_add_args(parser)
    logger.debug("added method args to parser.")

    _add_base_args(parser, default_args)
    logger.debug("added base args to parser.")

    args = parser.parse_args()
    write_demo = args.write_demo
    args = _process_gather_stats_args(args)
    logger.debug("processing args finished.")
    return args, write_demo


def _parse_general_args(parser, default_args):
    parser.add_argument(
        "--action",
        type=str,
        default=default_args.action,
        choices=default_args.actions,
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
        "--logging_level",
        type=int,
        default=default_args.logging_level,
    )

    args, _ = parser.parse_known_args()

    if args.assert_device:
        assert jax.device_count() > 0, "jax devices are not available"

    if args.disable_jit:
        logger.info("jit is disabled.")
        jax.config.update("jax_disable_jit", True)

    if args.dry_run:
        jax.config.update("jax_log_compiles", True)
        # jax.config.update('jax_platform_name', 'cpu')

    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)])

    logging.getLogger("source.utils").setLevel(args.logging_level)
    logging.getLogger("source.driver_helpers").setLevel(args.logging_level)
    logging.getLogger("source.explanation_methods.noise_interpolation").setLevel(
        args.logging_level
    )
    logging.getLogger("source.consistency_measures").setLevel(args.logging_level)
    logging.getLogger("source.operations").setLevel(args.logging_level)
    logging.getLogger("__main__").setLevel(args.logging_level)

    logger.debug("added general args to parser.")
    logger.debug(f"args: {args}")

    return args


def _add_base_args(parser, default_args):
    parser.add_argument(
        "--no_demo",
        action="store_false",
        dest="write_demo",
        default=default_args.write_demo,
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


def _process_gather_stats_args(args):
    os.makedirs(args.save_raw_data_dir, exist_ok=True)
    os.makedirs(args.save_metadata_dir, exist_ok=True)
    logger.debug("created the save directories.")

    dataset_query_func_switch[args.dataset](args)
    logger.debug("queried the dataset.")
    init_architecture_forward_switch[args.architecture](args)
    logger.debug("initialized the architecture.")

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
    logger.debug("initialized the stats.")

    method_args = _process_method_kwargs(args)
    logger.debug("updated the method and kwargs.")

    return method_args


def sample_demo(static_kwargs, dynamic_kwargs, meta_kwargs, stats):
    logger.info("sampling the demo.")
    method = meta_kwargs["method"]
    demo_stats = methods_switch[method].sample_demo(
        static_kwargs, dynamic_kwargs, meta_kwargs
    )
    stats.update(demo_stats)


def _make_loader(
    save_metadata_dir: str,
    pivot_indices: List[str],
    batch_size: int,
    measure_consistency_name: str,
    pivot_column: str = "alpha_mask_value",
    prefetch_factor=4,
):
    input_shape, merged_metadata = safely_load_metadata(
        save_metadata_dir,
        pivot_indices,
        pivot_column,
    )

    keys, merged_metadata = get_metadata_iterator(
        pivot_indices,
        measure_consistency_name,
        pivot_column,
    )

    index_iterator = get_index_iterator(merged_metadata)

    def _generator():
        for indices, paths in zip(*merged_metadata):
            sample = {}
            for key, path_batch in zip(keys, paths):
                sample[key] = np.stack(path_batch.apply(np.load))

            index = indices[0]  # the rest are the same
            index = {k: index[i] for i, k in index_iterator}
            yield {**sample, **index}

    indices_signature = {
        k: tf.TensorSpec(shape=(), dtype=tf.int32) for k in merged_metadata.index.names
    }
    samples_signature = {
        k: tf.TensorSpec(shape=input_shape, dtype=tf.float32) for k in keys
    }
    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature={
            **samples_signature,
            **indices_signature,
        },
    )
    iterator = dataset.batch(batch_size).prefetch(prefetch_factor).as_numpy_iterator()
    return iterator


def get_index_iterator(merged_metadata):
    assert isinstance(
        merged_metadata, tuple
    ), f"merged_metadata must be a tuple, got {type(merged_metadata)}"
    temp_metadata = merged_metadata[0]
    index_iterator = enumerate(temp_metadata.index.names)
    return index_iterator


def get_metadata_iterator(pivot_indices, measure_consistency_name, pivot_column):
    keys, merged_metadata_tuple = filter_relevant_parts(
        measure_consistency_name,
        merged_metadata_tuple,
    )

    pivot(
        pivot_indices,
        pivot_column,
        merged_metadata_tuple,
    )

    make_iterator(merged_metadata_tuple)

    return keys, merged_metadata_tuple


def make_iterator(merged_metadata_tuple):
    for i, _ in enumerate(merged_metadata_tuple):
        merged_metadata_tuple[i] = merged_metadata_tuple[i].iterrows()


def pivot(pivot_indices, pivot_column, merged_metadata_tuple):
    for i, metadata in enumerate(merged_metadata_tuple):
        # pivot table to get a dataframe with pivot_column as columns
        merged_metadata_tuple[i] = metadata.pivot(
            index=pivot_indices, columns=pivot_column, values="data_path"
        )
        # sort based on pivot_column
        merged_metadata_tuple[i] = merged_metadata_tuple[i].sort_index(axis=1)


def filter_relevant_parts(measure_consistency_name, merged_metadata):
    if measure_consistency_name == ConsistencyMeasures.cosine_distance:
        meanx2_metadata = merged_metadata[
            (merged_metadata["stream_name"] == "vanilla_grad_mask")
            & (merged_metadata["stream_statistic"] == "meanx2")
        ]
        keys = "meanx2"
        return keys, (meanx2_metadata,)

    elif measure_consistency_name == ConsistencyMeasures.dssim:
        meanx2_metadata = merged_metadata[
            (merged_metadata["stream_name"] == "vanilla_grad_mask")
            & (merged_metadata["stream_statistic"] == "meanx2")
        ]
        meanx_metadata = merged_metadata[
            (merged_metadata["stream_name"] == "vanilla_grad_mask")
            & (merged_metadata["stream_statistic"] == "meanx")
        ]
        keys = ("meanx", "meanx2")
        return keys, (meanx_metadata, meanx2_metadata)


def safely_load_metadata(save_metadata_dir, pivot_indices, pivot_column):
    merged_metadata_path = os.path.join(save_metadata_dir, f"merged_metadata.csv")
    assert os.path.exists(
        merged_metadata_path
    ), f"Could not find the merged metadata file in {save_metadata_dir}."
    merged_metadata = pd.read_csv(merged_metadata_path)
    logger.debug(f"loaded the merged metadata from {merged_metadata_path}.")

    assert "data_path" in merged_metadata.columns, (
        f"Could not find data_path column in {merged_metadata_path}. "
        f"Make sure the metadata file contains a column named data_path"
    )

    assert "input_shape" in merged_metadata.columns, (
        f"Could not find input_shape column in {merged_metadata_path}. "
        f"Make sure the metadata file contains a column named input_shape"
        f"or pass input_shape as an argument to loader_from_metadata"
    )
    input_shape = merged_metadata["input_shape"].iloc[0]
    input_shape = input_shape.replace("(", "").replace(")", "").split(",")
    input_shape = tuple(map(int, input_shape))
    logger.debug(f"found input_shape {input_shape} in the metadata file.")

    assert isinstance(
        input_shape, tuple
    ), f"input_shape must be a tuple, got {type(input_shape)}"
    if len(input_shape) == 4:
        assert (
            input_shape[0] == 1
        ), f"input_shape must have 4 dimensions (1, H, W, C), got {input_shape}"
        input_shape = input_shape[1:]
    else:
        assert (
            len(input_shape) == 3
        ), f"input_shape must have 3 dimensions (H, W, C), got {len(input_shape)}"

    assert pivot_column in merged_metadata.columns, (
        f"Could not find alpha_mask_value column in {merged_metadata_path}. "
        f"Make sure the metadata file contains a column named alpha_mask_value"
    )
    assert all(
        (pivot_index in merged_metadata.columns) for pivot_index in pivot_indices
    ), f"Could not find pivot_index columns {pivot_indices} in {merged_metadata.columns}. "

    num_distinct_values = len(merged_metadata[pivot_column].unique())
    assert num_distinct_values > 1, (
        f"Could not find more than 1 {pivot_column} in {merged_metadata_path}. "
        f"Make sure the metadata file contains more than 1 {pivot_column} to compute consistency."
    )
    logger.debug(f"found {num_distinct_values} alphas in the metadata file.")
    input_shape = (num_distinct_values, *input_shape)
    return input_shape, merged_metadata


def _process_method_kwargs(args):
    method_cls = methods_switch[args.method]
    args = method_cls.process_args(args)
    logger.debug("processed the method args.")
    return args


def save_gather_stats_data(save_raw_data_dir, stats):
    path_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    get_npy_file_path = lambda key: os.path.join(
        save_raw_data_dir, f"{path_prefix}.{key}.npy"
    )

    # temporary metadata
    npy_file_paths = []
    stream_name = []
    stream_statistic = []
    metadata = {}

    logger.debug("updating metadata experiment keys.")
    metadata["path_prefix"] = path_prefix

    logger.debug("saving the raw data and updating metadata sample keys.")
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

    logger.info(f"saved the raw data to {get_npy_file_path('*')}")

    return metadata


def save_gather_stats_metadata(save_metadata_dir, metadata):
    csv_file_name = f"{metadata['path_prefix']}.csv"
    metadata_file_path = os.path.join(save_metadata_dir, csv_file_name)
    metadata["metadata_file_path"] = metadata_file_path

    del metadata["monitored_statistic_source_key"]
    del metadata["monitored_statistic_key"]
    del metadata["batch_index_key"]
    del metadata["stats"]

    metadata = {k: w for k, w in metadata.items() if w != None}

    metadata["projection_index"] = int(metadata["projection_index"])
    metadata["input_shape"] = str(metadata["input_shape"])

    # convert metadata from dict to dataframe and save
    dataframe = pd.DataFrame(metadata)
    dataframe.to_csv(metadata_file_path, index=False)
    logger.info(f"saved the correspoding meta data to {metadata_file_path}")


def save_consistency(
    save_metadata_dir,
    metadata,
    pivot_column,
    consistency_measure_name,
):
    csv_file_name = f"consistency_{consistency_measure_name}_{pivot_column}.csv"
    metadata_file_path = os.path.join(save_metadata_dir, csv_file_name)

    # convert metadata from dict to dataframe and save
    logger.debug(str({k: v.shape for k, v in metadata.items()}))
    dataframe = pd.DataFrame(metadata)
    dataframe.to_csv(metadata_file_path, index=False)
    logger.info(f"saved the correspoding meta data to {metadata_file_path}")
