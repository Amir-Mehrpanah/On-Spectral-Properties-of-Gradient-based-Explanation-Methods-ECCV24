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
import jax.dlpack
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


sys.path.append(os.getcwd())
from source.configs import DefaultArgs
from source.data_manager import (
    CBIS_DDSM_CraftedDecoder,
    Food101CraftedDecoder,
    curated_breast_imaging_ddsm_loader_from_metadata,
    food101_loader_from_metadata,
    imagenet_loader_from_metadata,
    query_curated_breast_imaging_ddsm,
    query_imagenet,
    query_food101,
    TypeOrNan,
)
from source.project_manager import load_experiment_metadata
from source.explanation_methods.noise_interpolation import NoiseInterpolation
from source.model_manager import init_resnet50_forward, init_resnet50_randomized_forward, init_vit_forward
from source.inconsistency_measures import (
    _measure_inconsistency_cosine_distance,
    _measure_inconsistency_DSSIM,
)
from source.utils import (
    Action,
    InconsistencyMeasures,
    Switch,
    Stream,
    StreamNames,
    Statistics,
    debug_nice,
)

logger = logging.getLogger(__name__)


def gpu_preallocation():
    # Needed for TensorFlow and JAX to coexist in GPU memory.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized.
            print(e)


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
dataset_query_func_switch.register(
    "food101",
    query_food101,
)
dataset_query_func_switch.register(
    "curated_breast_imaging_ddsm",
    query_curated_breast_imaging_ddsm,
)
init_architecture_forward_switch.register(
    "resnet50",
    init_resnet50_forward,
)
init_architecture_forward_switch.register(
    "resnet50-randomized",
    init_resnet50_randomized_forward,
)
init_architecture_forward_switch.register(
    "vit_b_16_224",
    init_vit_forward,
)

def base_parser(parser, default_args: DefaultArgs):
    args = _parse_general_args(parser, default_args)

    if args.action == Action.gather_stats:
        action_args, write_demo = _parse_gather_stats_args(
            parser,
            default_args,
        )
        driver_args = argparse.Namespace(
            action=args.action,
            write_demo=write_demo,
            save_raw_data_dir=args.save_raw_data_dir,
            save_metadata_dir=args.save_metadata_dir,
            skip_data=args.skip_data,
        )
    elif args.action == Action.compute_entropy:
        action_args = argparse.Namespace()
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    elif args.action == Action.compute_inconsistency:
        action_args = _parse_measure_inconsistency_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    elif args.action == Action.merge_stats:
        action_args = _merge_stats_args(parser)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    elif args.action == Action.compute_integrated_grad:
        action_args = _parse_integrated_grad_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
            save_raw_data_dir=args.save_raw_data_dir,
        )
    elif args.action == Action.compute_accuracy_at_q:
        action_args = _parse_compute_accuracy_at_q_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    else:
        raise NotImplementedError("other actions are not implemented")

    return driver_args, action_args


def _parse_compute_accuracy_at_q_args(parser, default_args):
    parser.add_argument(
        "--save_file_name_prefix",
        type=str,
        default="accuracy_at_q",
    )
    parser.add_argument(
        "--q",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--q_direction",
        type=str,
        required=True,
        choices=["deletion", "insertion"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=default_args.architectures,
    )
    parser.add_argument(
        "--input_shape",
        nargs=3,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output_layer",
        type=str,
        default=default_args.output_layer,
        choices=default_args.output_layers,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args.batch_size,
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=default_args.prefetch_factor,
    )
    parser.add_argument(
        "--save_temp_base_dir",
        type=str,
        default=default_args.save_temp_base_dir,
    )
    parser.add_argument(
        "--glob_path",
        type=str,
        default="*.csv",
    )
    parser.add_argument(
        "--q_baseline_mask",
        type=str,
        default="blur",
        choices=["blur", "black"],
    )
    parser.add_argument(
        "--filter_alpha_prior",
        type=TypeOrNan(str),
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_args.dataset,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=default_args.num_classes,
    )

    args, _ = parser.parse_known_args()

    assert (args.dataset == "imagenet") or (
        args.dataset_dir is not None
    ), "dataset_dir must be provided for datasets other than imagenet."

    input_shape = tuple(args.input_shape)

    sl_metadata = load_experiment_metadata(args.save_metadata_dir, args.glob_path)
    logger.debug(
        f"loaded metadata from {args.save_metadata_dir} with {args.glob_path} of shape {sl_metadata.shape}."
    )

    ids = sl_metadata["stream_name"] == "vanilla_grad_mask"
    if args.filter_alpha_prior:
        logger.debug(f"filtering alpha_mask_value to {args.filter_alpha_prior}.")
        ids = ids & (sl_metadata["alpha_mask_value"] == args.filter_alpha_prior)
    else:
        logger.debug(f"no filter for alpha_mask_value except vanilla_grad_mask.")

    sl_metadata = sl_metadata[ids]
    logger.debug(f"filtered index {sl_metadata.index} columns {sl_metadata.columns}.")
    sl_metadata = sl_metadata.reset_index(drop=True)
    logger.debug(f"filtered metadata to shape {sl_metadata.shape}.")

    if args.dataset == "imagenet":
        slq_dataloader = imagenet_loader_from_metadata(
            sl_metadata,
            args.q,
            args.q_direction,
            baseline=args.q_baseline_mask,
            input_shape=input_shape,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch_factor,
        )
    elif args.dataset == "food101":
        # sort based on image_index
        sl_metadata = sl_metadata.sort_values("image_index")
        slq_dataloader = food101_loader_from_metadata(
            sl_metadata,
            args.q,
            args.q_direction,
            input_shape=input_shape,
            baseline=args.q_baseline_mask,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch_factor,
            dataset_dir=args.dataset_dir,
        )
    elif args.dataset == "curated_breast_imaging_ddsm":
        sl_metadata = sl_metadata.sort_values("image_index")
        slq_dataloader = curated_breast_imaging_ddsm_loader_from_metadata(
            sl_metadata,
            args.q,
            args.q_direction,
            input_shape=input_shape,
            baseline=args.q_baseline_mask,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch_factor,
            dataset_dir=args.dataset_dir,
        )
    else:
        raise NotImplementedError("other datasets are not implemented")

    init_architecture_forward_switch[args.architecture](args)

    return argparse.Namespace(
        slq_dataloader=slq_dataloader,
        sl_metadata=sl_metadata,
        forward=args.forward[0],
        params=args.params[0],
        save_file_name_prefix=args.save_file_name_prefix,
        q=args.q,
        q_direction=args.q_direction,
        q_baseline_mask=args.q_baseline_mask,
    )


def _merge_stats_args(parser):
    parser.add_argument(
        "--glob_path",
        type=str,
        default="*.csv",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="merged_metadata.csv",
    )
    args, _ = parser.parse_known_args()
    return args


def _parse_integrated_grad_args(parser, default_args):
    parser.add_argument(
        "--alpha_mask_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--alpha_prior",
        type=float,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--projection_type",
        type=str,
        required=True,
        choices=["label", "random", "prediction", "static"],
    )
    parser.add_argument(
        "--projection_top_k",
        type=TypeOrNan(int),
        default=None,
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=4,
        default=default_args.input_shape,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mean_rgb",
        type=float,
        default=None,
        nargs=1,
    )
    parser.add_argument(
        "--std_rgb",
        type=float,
        default=None,
        nargs=1,
    )
    args, _ = parser.parse_known_args()

    args.random_access_dataset = None
    if "_i_" in args.alpha_mask_name:
        if args.dataset == "food101":
            assert (
                (args.dataset_dir is not None)
                and (args.mean_rgb is not None)
                and (args.std_rgb is not None)
            ), "dataset_dir, mean_rgb and std_rgb must be provided for food101 dataset."
            del args.dataset
            
            food_dataset = tfds.data_source(
                "food101",
                split="validation",
                data_dir=args.dataset_dir,
                download=False,
                decoders={
                    "image": Food101CraftedDecoder(
                        args.input_shape,
                        args.mean_rgb,
                        args.std_rgb,
                    )
                },
            )
            args.random_access_dataset = food_dataset
        elif args.dataset == "curated_breast_imaging_ddsm":
            assert (
                (args.dataset_dir is not None)
                and (args.mean_rgb is not None)
                and (args.std_rgb is not None)
            ), "dataset_dir, mean_rgb and std_rgb must be provided for curated_breast_imaging_ddsm dataset."
            del args.dataset
            
            cbis_dataset = tfds.data_source(
                "curated_breast_imaging_ddsm",
                split="validation",
                data_dir=args.dataset_dir,
                download=False,
                decoders={
                    "image": CBIS_DDSM_CraftedDecoder(
                        args.input_shape,
                        args.mean_rgb,
                        args.std_rgb,
                    )
                },
            )
            args.random_access_dataset = cbis_dataset
    return args


def _parse_measure_inconsistency_args(parser, default_args):
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args.batch_size,
    )
    parser.add_argument(
        "--pivot_indices",
        nargs="+",
        type=str,
        default=default_args.pivot_indices,
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
        "--inconsistency_measure",
        type=str,
        required=True,
        choices=default_args.inconsistency_measures,
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=default_args.c1,
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=default_args.c2,
    )

    args, _ = parser.parse_known_args()
    data_loader = _make_loader(
        args.save_metadata_dir,
        args.pivot_indices,
        args.batch_size,
        args.inconsistency_measure,
        args.pivot_column,
        prefetch_factor=args.prefetch_factor,
    )
    inconsistency_measure_func = get_inconsistency_measure(args)

    return argparse.Namespace(
        data_loader=data_loader,
        pivot_column=args.pivot_column,
        inconsistency_measure=inconsistency_measure_func,
        inconsistency_measure_name=args.inconsistency_measure,
    )


def get_inconsistency_measure(args):
    if args.inconsistency_measure == InconsistencyMeasures.cosine_distance:
        inconsistency_measure_func = _measure_inconsistency_cosine_distance(
            downsampling_factor=args.downsampling_factor,
            downsampling_method=jax.image.ResizeMethod.LINEAR,
        )
    elif args.inconsistency_measure == InconsistencyMeasures.dssim:
        inconsistency_measure_func = _measure_inconsistency_DSSIM(
            downsampling_factor=args.downsampling_factor,
            downsampling_method=jax.image.ResizeMethod.LINEAR,
            c1=args.c1,
            c2=args.c2,
        )
    else:
        raise NotImplementedError("other inconsistency measures are not implemented")
    logger.debug(f"inconsistency_measure_func set to {inconsistency_measure_func}")
    inconsistency_measure_func = inconsistency_measure_func.concretize()
    inconsistency_measure_func = jax.jit(inconsistency_measure_func)
    return inconsistency_measure_func


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

    _add_gather_stats_base_args(parser, default_args)
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
        "--skip_data",
        type=str,
        nargs="+",
        default=None,
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
    logging.getLogger("source.project_manager").setLevel(args.logging_level)
    logging.getLogger("source.driver_helpers").setLevel(args.logging_level)
    logging.getLogger("source.explanation_methods.noise_interpolation").setLevel(
        args.logging_level
    )
    logging.getLogger("source.inconsistency_measures").setLevel(args.logging_level)
    logging.getLogger("source.data_manager").setLevel(args.logging_level)
    logging.getLogger("source.model_manager").setLevel(args.logging_level)
    logging.getLogger("source.operations").setLevel(args.logging_level)
    logging.getLogger("__main__").setLevel(args.logging_level)

    logger.debug("added general args to parser.")
    logger.debug(f"args: {args}")

    return args


def _add_gather_stats_base_args(parser, default_args):
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
        "--save_temp_base_dir",
        type=str,
        default=default_args.save_temp_base_dir,
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
        "--mean_rgb",
        nargs=1,
        type=float,
        default=np.nan,
    )
    parser.add_argument(
        "--std_rgb",
        nargs=1,
        type=float,
        default=np.nan,
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
        "--layer_randomization",
        type=str,
        default=None,
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

    args.input_shape = tuple(args.input_shape)
    if not np.isnan(args.mean_rgb).any():
        assert not np.isnan(
            args.std_rgb
        ).any(), "mean_rgb and std_rgb must be both provided."
        args.mean_rgb = jnp.array(args.mean_rgb)
        args.std_rgb = jnp.array(args.std_rgb)

    dataset_query_func_switch[args.dataset](args)
    logger.debug("queried the dataset.")

    init_architecture_forward_switch[args.architecture](args)
    logger.debug("initialized the architecture.")

    if args.monitored_statistic != Statistics.meanx2:
        raise NotImplementedError("other stats are not implemented")

    if args.monitored_stream != StreamNames.vanilla_grad_mask:
        raise NotImplementedError("other streams must be implemented")

    args.monitored_statistic_key = Stream(
        args.monitored_stream,
        Statistics.abs_delta,
    )
    args.monitored_statistic_source_key = Stream(
        args.monitored_stream,
        args.monitored_statistic,
    )
    args.batch_index_key = Stream(
        StreamNames.batch_index,
        Statistics.none,
    )
    args.stats = {
        args.monitored_statistic_source_key: jnp.zeros(shape=args.input_shape),
        # Stream(
        #     StreamNames.results_at_projection,
        #     Statistics.meanx,
        # ): jnp.zeros(shape=()),
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
            # args.stats[
            #     Stream(
            #         StreamNames.results_at_projection,
            #         Statistics.meanx2,
            #     )
            # ] = jnp.zeros(shape=())

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
    measure_inconsistency_name: str,
    pivot_column: str,
    prefetch_factor: int,
):
    input_shape, merged_metadata_tuple = safely_load_metadata(
        save_metadata_dir,
        pivot_indices,
        pivot_column,
    )

    sample_keys, merged_metadata_tuple = prepare_metadata(
        pivot_indices,
        measure_inconsistency_name,
        pivot_column,
        merged_metadata_tuple,
    )

    index_keys = get_index_keys(merged_metadata_tuple)
    merged_metadata_tuple = make_iterator(merged_metadata_tuple)

    def _generator():
        for items in zip(*merged_metadata_tuple):
            sample = []
            for indices, paths in items:
                sample.append(np.stack(paths.apply(np.load)))
            sample = tuple(sample)
            index = {k: v for k, v in zip(index_keys, indices)}
            yield {"data": sample, **index}

    indices_signature, samples_signature = get_output_signatures(
        input_shape,
        sample_keys,
        index_keys,
    )

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature={
            **samples_signature,
            **indices_signature,
        },
    )
    iterator = dataset.batch(batch_size).prefetch(prefetch_factor).as_numpy_iterator()
    return iterator


def get_output_signatures(input_shape, sample_keys, index_keys):
    indices_signature = {k: tf.TensorSpec(shape=(), dtype=tf.int32) for k in index_keys}
    samples_signature = {
        "data": (tf.TensorSpec(shape=input_shape, dtype=tf.float32),) * len(sample_keys)
    }

    logger.debug(
        f"samples_signature: {samples_signature} \n"
        f"indices_signature: {indices_signature}"
    )
    return indices_signature, samples_signature


def get_index_keys(merged_metadata):
    assert isinstance(
        merged_metadata, tuple
    ), f"merged_metadata must be a tuple, got {type(merged_metadata)}"

    return merged_metadata[0].index.names


def prepare_metadata(
    pivot_indices,
    measure_inconsistency_name,
    pivot_column,
    merged_metadata,
):
    sample_keys, merged_metadata_tuple = filter_relevant_parts(
        measure_inconsistency_name,
        merged_metadata,
    )

    merged_metadata_tuple = pivot_metadata(
        pivot_indices,
        pivot_column,
        merged_metadata_tuple,
    )

    assert isinstance(
        merged_metadata_tuple, tuple
    ), f"merged_metadata_tuple must be a tuple, got {type(merged_metadata_tuple)}"
    assert isinstance(
        sample_keys, tuple
    ), f"sample_keys must be a tuple, got {type(sample_keys)}"

    return sample_keys, merged_metadata_tuple


def make_iterator(merged_metadata_tuple):
    output = []
    logger.debug(f"making iterator for {debug_nice(merged_metadata_tuple)}")
    for metadata in merged_metadata_tuple:
        output.append(metadata.iterrows())
    merged_metadata_tuple = tuple(output)
    return merged_metadata_tuple


def pivot_metadata(pivot_indices, pivot_column, merged_metadata_tuple):
    output = []
    for metadata in merged_metadata_tuple:
        # pivot table to get a dataframe with pivot_column as columns
        output.append(
            metadata.pivot(
                index=pivot_indices, columns=pivot_column, values="data_path"
            )
        )
        # sort based on pivot_column
        output[-1] = output[-1].sort_index(axis=1)

    # convert to a list of tuples
    merged_metadata_tuple = tuple(output)
    return merged_metadata_tuple


def filter_relevant_parts(measure_inconsistency_name, merged_metadata):
    if measure_inconsistency_name == InconsistencyMeasures.cosine_distance:
        meanx2_metadata = merged_metadata[
            (merged_metadata["stream_name"] == "vanilla_grad_mask")
            & (merged_metadata["stream_statistic"] == "meanx2")
        ]
        keys = ("meanx2",)
        return keys, (meanx2_metadata,)

    elif measure_inconsistency_name == InconsistencyMeasures.dssim:
        meanx2_metadata = merged_metadata[
            (merged_metadata["stream_name"] == StreamNames.vanilla_grad_mask)
            & (merged_metadata["stream_statistic"] == Statistics.meanx2)
        ]
        meanx_metadata = merged_metadata[
            (merged_metadata["stream_name"] == StreamNames.vanilla_grad_mask)
            & (merged_metadata["stream_statistic"] == Statistics.meanx)
        ]
        assert len(meanx2_metadata) == len(meanx_metadata), (
            f"meanx2_metadata and meanx_metadata must have"
            f" the same length, got {len(meanx2_metadata)}"
            f" and {len(meanx_metadata)}"
        )
        keys = (Statistics.meanx, Statistics.meanx2)
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
        f"Make sure the metadata file contains more than 1 {pivot_column} to compute inconsistency."
    )
    logger.debug(f"found {num_distinct_values} alphas in the metadata file.")
    input_shape = (num_distinct_values, *input_shape)
    return input_shape, merged_metadata


def _process_method_kwargs(args):
    method_cls = methods_switch[args.method]
    args = method_cls.process_args(args)
    logger.debug("processed the method args.")
    return args


def save_gather_stats_data(save_raw_data_dir, skip_data, stats):
    # with large number of synchronized parallel jobs,
    # the same path_prefix might be generated.
    # To avoid this, we add a random number
    # to the path_prefix to reduce the chance of collision.
    rnd = np.random.randint(0, 10000)
    path_prefix = datetime.now().strftime(f"%m%d_%H%M%S%f_{rnd:04d}")

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
        file_path = f"{key.name}.{key.statistic}"
        npy_file_path = get_npy_file_path(file_path)
        if skip_data and key.name in skip_data:
            logger.info(f"skipped writing {npy_file_path}")
            continue

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


def save_inconsistency(
    save_metadata_dir,
    metadata,
    pivot_column,
    inconsistency_measure_name,
):
    csv_file_name = f"inconsistency_{inconsistency_measure_name}_{pivot_column}.csv"
    metadata_file_path = os.path.join(save_metadata_dir, csv_file_name)

    # convert metadata from dict to dataframe and save
    logger.debug(f"saving the inconsistency metadata {debug_nice(metadata)}")
    dataframe = pd.DataFrame(metadata)
    dataframe.to_csv(metadata_file_path, index=False)
    logger.info(f"saved the correspoding meta data to {metadata_file_path}")


def tf_to_jax(x):
    x = jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(x))
    x = jax.device_put(x, jax.devices()[0])
    return x


def compute_accuracy_at_q(
    save_metadata_dir,
    sl_metadata,
    save_file_name_prefix,
    q,
    q_direction,
    q_baseline_mask,
    forward,
    params,
    slq_dataloader,
):
    preds = []
    actual_qs = []
    total_steps = sl_metadata.shape[0]
    for i, batch in enumerate(slq_dataloader):
        logger.debug(f"batch shape: {batch['masked_image'].shape}")
        masked_image = tf_to_jax(batch["masked_image"])
        logger.debug(
            f"jax default device: {jax.devices()} "
            f"masked_image: {masked_image.device_buffer.device()}"
        )
        logger.debug(f"batch: {i} of {total_steps}//batch_size time: {datetime.now()}")
        logits = forward(params,masked_image)
        logits = logits.argmax(axis=1)
        preds.append(logits == batch["label"])
        actual_qs.append(batch["actual_q"])

    # convert preds to dataframe
    preds = pd.DataFrame(
        {
            "preds": np.concatenate(preds, axis=0),
            "actual_q": np.concatenate(actual_qs, axis=0),
        },
    )
    preds["q"] = q
    preds["q_direction"] = q_direction
    preds["q_baseline_mask"] = q_baseline_mask

    logger.debug(f"preds shape: {preds.shape} (q results)")
    logger.debug(
        f"sl_metadata shape: {sl_metadata.shape} before concatenation of q results"
    )

    sl_metadata = pd.concat([sl_metadata, preds], axis=1)
    logger.debug(
        f"sl_metadata shape: {sl_metadata.shape} after concatenation of q results"
    )

    file_name = f"{save_file_name_prefix}_{q}.csv"
    sl_metadata.to_csv(os.path.join(save_metadata_dir, file_name), index=False)
