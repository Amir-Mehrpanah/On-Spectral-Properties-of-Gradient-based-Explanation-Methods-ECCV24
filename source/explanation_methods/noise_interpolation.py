from source.utils import (
    Statistics,
    Stream,
    StreamNames,
    AbstractFunction,
    pattern_generator,
    debug_nice,
)
from source import neighborhoods, explainers, operations
from source.model_manager import forward_with_projection
from source.data_manager import minmax_normalize, _bool, TypeOrNan
import argparse
import copy
from functools import partial
import logging
import jax
import json
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Tuple
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


class NoiseInterpolation:
    @staticmethod
    def sampler(
        key,
        forward,
        projection,
        alpha_mask,
        image,
        baseline_mask,
        combination_fn,
        explainer_fn,
        demo=False,
    ):
        if isinstance(baseline_mask, Callable):
            baseline_mask = baseline_mask(key=key)
        if isinstance(projection, Callable):
            projection = projection(key=key)
        if isinstance(alpha_mask, Callable):
            alpha_mask = alpha_mask(key=key)

        combination_mask = combination_fn(
            source_mask=image,
            target_mask=baseline_mask,
            alpha_mask=alpha_mask,
        )

        _forward = partial(
            forward_with_projection,
            projection=projection,
            forward=forward,
        )

        (
            vanilla_grad_mask,
            log_probs,
        ) = explainer_fn(
            forward=_forward,
            inputs=combination_mask,
            alpha_mask=alpha_mask,
        )

        if demo:
            return {
                Stream(
                    StreamNames.vanilla_grad_mask, Statistics.none
                ): vanilla_grad_mask,
                Stream(StreamNames.log_probs, Statistics.none): log_probs,
                Stream(StreamNames.image, Statistics.none): image,
                Stream("combination_mask", Statistics.none): combination_mask,
                Stream("projection", Statistics.none): projection,
                Stream("alpha_mask", Statistics.none): alpha_mask,
                Stream("baseline_mask", Statistics.none): baseline_mask,
                Stream("combination_fn", Statistics.none): combination_fn,
                Stream("explainer_fn", Statistics.none): explainer_fn,
            }
        return {
            StreamNames.vanilla_grad_mask: vanilla_grad_mask,
            StreamNames.log_probs: log_probs,
        }

    static_sampler = AbstractFunction(sampler.__func__)
    sampler_args = list(static_sampler.params.keys())
    assert sampler_args[0] == "key", "key must be the first arg of sampler"
    sampler_args.remove("key")

    def inplace_add_args(self, base_parser):
        base_parser.add_argument(
            "--alpha_mask_type",
            type=str,
            required=True,
            nargs="+",
            choices=[
                "static",
                "scalar_uniform",
                # "image_onehot-7x7",
                # "image_onehot-10x10",
                # "image_onehot-14x14",
                "image_bernoulli-7x7",
                "image_bernoulli-10x10",
                "image_bernoulli-14x14",
            ],
        )
        base_parser.add_argument(
            "--alpha_mask_value",
            type=TypeOrNan(type=float),
            nargs="*",
            default=[np.nan],
        )
        base_parser.add_argument(
            "--projection_type",
            type=str,
            nargs="+",
            required=True,
            choices=["label", "random", "prediction", "static"],
        )
        base_parser.add_argument(
            "--projection_top_k",
            type=TypeOrNan(int),
            nargs="*",
            default=[np.nan],
        )
        base_parser.add_argument(
            "--projection_index",
            type=TypeOrNan(int),
            nargs="*",
            default=[np.nan],
        )
        base_parser.add_argument(
            "--projection_distribution",
            type=TypeOrNan(str),
            choices=[None, "uniform", "categorical", "delta"],
            default=["delta"],
            nargs="*",
        )
        base_parser.add_argument(
            "--baseline_mask_type",
            type=str,
            required=True,
            choices=[
                "static",
                "gaussian",
                "gaussian-0.1",
                "gaussian-0.3",
                "gaussian-0.5",
                "gaussian-0.7",
                "gaussian-1.0",
            ],
            nargs="+",
        )
        base_parser.add_argument(
            "--baseline_mask_value",
            type=TypeOrNan(float),
            default=[np.nan],
            nargs="*",
        )
        base_parser.add_argument(
            "--combination_fn",
            type=str,
            required=True,
            nargs="+",
            choices=[
                "convex",
                "additive",
                "damping",
            ],
        )
        base_parser.add_argument(
            "--explainer_fn",
            type=str,
            nargs="+",
            default=["vanilla_grad"],
            choices=[
                "vanilla_grad",
                "finite_difference",
            ],
        )

    @classmethod
    def process_args(cls, args):
        mixed_args = cls.extract_mixed_args(args)
        mixed_pattern = cls.extract_mixed_pattern(args.args_pattern, mixed_args)
        mixed_args = cls.maybe_broadcast_shapes(mixed_pattern, mixed_args)
        num_samplers = cls.compute_num_samplers(mixed_args, mixed_pattern)
        if logger.isEnabledFor(logging.INFO):
            cls.pretty_print_args(mixed_args)

        if logger.isEnabledFor(logging.DEBUG):
            nice_mixed_args = debug_nice(mixed_args, max_depth=1)
            logger.debug(
                f"mixed_pattern: {mixed_pattern}\nmixed_args: {nice_mixed_args}"
            )

        combined_mixed_args = pattern_generator(mixed_pattern, mixed_args)
        combined_mixed_args = map(cls._process_logics, combined_mixed_args)
        combined_mixed_args = map(cls._process_args, combined_mixed_args)
        splitted_args = cls._split_args_dicts(
            combined_mixed_args,
            args_state=args.args_state,
        )
        samplers_and_kwargs = cls.sampler_generator(splitted_args)

        return argparse.Namespace(
            samplers_and_kwargs=samplers_and_kwargs,
            num_samplers=num_samplers,
        )

    @staticmethod
    def compute_num_samplers(mixed_args, mixed_pattern):
        num_samplers = 1
        inverted_mixed_pattern = {v: k for k, v in mixed_pattern.items()}
        unique_pattern_values = list(set(inverted_mixed_pattern.values()))
        for unique_pattern_value in unique_pattern_values:
            num_samplers *= len(mixed_args[unique_pattern_value])
        return num_samplers

    @classmethod
    def sampler_generator(cls, splitted_args):
        for (
            combined_dynamic_kwargs,
            combined_static_kwargs,
            combined_meta_kwargs,
        ) in splitted_args:
            combined_dynamic_kwargs = cls._sort_dynamic_kwargs(combined_dynamic_kwargs)
            vmap_axis = (0,) + tuple(
                None for _ in combined_dynamic_kwargs
            )  # 0 for key, None for dynamic args
            sampler = cls._create_sampler(
                combined_static_kwargs,
                vmap_axis,
            )

            yield sampler, combined_static_kwargs, combined_dynamic_kwargs, combined_meta_kwargs

    @classmethod
    def pretty_print_args(cls, mixed_args: argparse.Namespace):
        pretty_kwargs = copy.deepcopy(mixed_args)
        pretty_kwargs["image"] = f"{len(pretty_kwargs['image'])} number of images"
        pretty_kwargs["forward"] = f"forward of len {len(pretty_kwargs['forward'])}"

        temp_stats = f"[{len(pretty_kwargs['stats'])} stats of len {len(pretty_kwargs['stats'][0])}]"
        pretty_kwargs["stats"] = temp_stats
        pretty_kwargs["projection_index"] = [
            int(v) if not np.isnan(v) else v for v in pretty_kwargs["projection_index"]
        ]
        logger.info(
            f"experiment args:\n{debug_nice(pretty_kwargs)}",
        )

    @classmethod
    def _process_args(cls, args_dict):
        args_dict = cls._process_projection(args_dict)
        args_dict = cls._process_baseline_mask(args_dict)
        args_dict = cls._process_alpha_mask(args_dict)
        args_dict = cls._process_combination_fn(args_dict)
        args_dict = cls._process_explainer_fn(args_dict)

        for arg in cls.sampler_args:
            assert (
                arg in args_dict
            ), "processed args_dict must contains all args that expected by sampler except key"
        assert not "key" in args_dict, "key is a reserved word"
        return args_dict

    @staticmethod
    def maybe_broadcast_shapes(pattern, values):
        pattern_values = list(pattern.values())
        pattern_keys = list(pattern.keys())
        temp_values = [values[k] for k in pattern_keys]
        len_values = [len(v) for v in temp_values]
        unique_pattern_values = list(set(pattern_values))
        for unique_pattern_value in unique_pattern_values:
            # find the max length of lists with the same pattern id
            temp_len_values = [
                len_values[i]
                for i, v in enumerate(pattern_values)
                if v == unique_pattern_value
            ]
            # assert that shapes are either 1 or the same as the max
            max_value = np.max(temp_len_values)
            assert all(
                v == 1 or v == max_value for v in temp_len_values
            ), f"lists with the same pattern id must have the same length or one {pattern}"

            # broadcast lists with the same pattern id to the max length
            for i, pattern_value in enumerate(pattern_values):
                if pattern_value == unique_pattern_value and len_values[i] != max_value:
                    len_factor = int(max_value / len_values[i])
                    pattern_key = pattern_keys[i]
                    # assumed that values[pattern_key] has only one element see assertion above
                    values[pattern_key] = [
                        values[pattern_key][0] for j in range(len_factor)
                    ]
                    logger.debug(
                        f"broadcasting {pattern_key} from {len_values[i]} to {max_value} according to {pattern[pattern_key]}",
                    )

        return values

    @classmethod
    def extract_mixed_pattern(cls, args_pattern, mixed_args):
        def inplace_infer(pattern, k1, k2):
            if k1 not in pattern:
                pattern[k1] = pattern[k2]

        if "method" not in args_pattern:
            args_pattern["method"] = "method"

        inplace_infer(args_pattern, "baseline_mask", "method")
        inplace_infer(args_pattern, "combination_fn", "method")
        inplace_infer(args_pattern, "explainer_fn", "method")
        inplace_infer(args_pattern, "projection", "method")
        inplace_infer(args_pattern, "alpha_mask", "method")
        inplace_infer(args_pattern, "image", "method")
        inplace_infer(args_pattern, "forward", "method")
        inplace_infer(args_pattern, "label", "image")
        inplace_infer(args_pattern, "input_shape", "forward")
        inplace_infer(args_pattern, "num_classes", "forward")
        inplace_infer(args_pattern, "architecture", "forward")
        inplace_infer(args_pattern, "output_layer", "forward")
        inplace_infer(args_pattern, "layer_randomization", "forward")
        inplace_infer(args_pattern, "seed", "method")
        inplace_infer(args_pattern, "stats", "method")
        inplace_infer(args_pattern, "demo", "method")
        inplace_infer(args_pattern, "dataset", "method")
        inplace_infer(args_pattern, "monitored_statistic_key", "method")
        inplace_infer(args_pattern, "batch_size", "method")
        inplace_infer(args_pattern, "max_batches", "method")
        inplace_infer(args_pattern, "min_change", "method")
        inplace_infer(args_pattern, "monitored_statistic_source_key", "method")
        inplace_infer(args_pattern, "batch_index_key", "method")

        mixed_pattern = {}
        for arg_name in mixed_args:
            pattern_proposal = [v for k, v in args_pattern.items() if k in arg_name]
            assert (
                len(pattern_proposal) == 1
            ), f"{arg_name} has the following proposals {pattern_proposal} according to the provided pattern {args_pattern} and cannot be uniquely identified"
            mixed_pattern[arg_name] = pattern_proposal[0]
        return mixed_pattern

    @classmethod
    def _create_sampler(cls, static_kwargs, vamp_axis=None):
        sampler = AbstractFunction(cls.sampler)(**static_kwargs).concretize()
        if vamp_axis is not None:
            sampler = jax.vmap(sampler, in_axes=vamp_axis)
        jax.jit(sampler)
        return sampler

    @classmethod
    def _sort_dynamic_kwargs(cls, dynamic_kwargs_dict):
        # sort dynamic args according to sampler args order
        dynamic_kwargs_dict = {
            arg: dynamic_kwargs_dict[arg]
            for arg in cls.sampler_args
            if arg in dynamic_kwargs_dict
        }
        return dynamic_kwargs_dict

    @staticmethod
    def extract_mixed_args(args):
        input_args = [
            "alpha_mask_type",
            "alpha_mask_value",
            "baseline_mask_type",
            "baseline_mask_value",
            "combination_fn",
            "explainer_fn",
            "projection_type",
            "projection_distribution",
            "projection_top_k",
            "projection_index",
            "label",
            "image",
            "forward",
            "architecture",
            "method",
            "output_layer",
            "layer_randomization",
            "dataset",
            "image_index",
            "image_path",
            "input_shape",
            "num_classes",
            "monitored_statistic_key",
            "seed",
            "batch_size",
            "max_batches",
            "min_change",
            "monitored_statistic_source_key",
            "batch_index_key",
            "stats",
        ]
        mixed_args = {}
        for arg_name in input_args:
            assert hasattr(
                args, arg_name
            ), f"method expects an arg that is not in the provided args: {arg_name}"
            item = getattr(args, arg_name)
            mixed_args[arg_name] = item if isinstance(item, list) else [item]

        mixed_args["demo"] = [False]
        return mixed_args

    @classmethod
    def _split_args_dicts(cls, combined_mixed_args, args_state):
        dynamic_keys = [k for k, v in args_state.items() if "dynamic" in v]
        static_keys = [k for k in cls.sampler_args if k not in dynamic_keys]
        meta_keys = [k for k, v in args_state.items() if "meta" in v]
        assert len(static_keys) + len(dynamic_keys) == len(
            cls.sampler_args
        ), f"static and dynamic keys must be equal to sampler args {cls.sampler_args}"

        for args_dict in combined_mixed_args:
            (
                temp_dynamic_kwargs,
                temp_static_kwargs,
                temp_meta_args,
            ) = cls._split_args_dict(
                args_dict,
                static_keys,
                dynamic_keys,
                meta_keys,
            )
            yield temp_dynamic_kwargs, temp_static_kwargs, temp_meta_args

    @staticmethod
    def _split_args_dict(args_dict, static_keys, dynamic_keys, meta_keys):
        static_kwargs = {}
        dynamic_kwargs = {}
        meta_kwargs = {}
        for k, v in args_dict.items():
            if k in static_keys:
                static_kwargs[k] = v
            elif k in dynamic_keys:
                dynamic_kwargs[k] = v
            if k in meta_keys or (not (k in static_keys)) and (not (k in dynamic_keys)):
                meta_kwargs[k] = v
        return dynamic_kwargs, static_kwargs, meta_kwargs

    @classmethod
    def _process_logics(
        cls,
        args_dict,
    ):
        assert len(args_dict["input_shape"]) == 4
        cls._process_logics_alpha_mask(args_dict)
        cls._process_logics_baseline_mask(args_dict)
        cls._process_logics_projection(args_dict)

        return args_dict

    @staticmethod
    def _process_logics_projection(args_dict):
        if args_dict["projection_type"] == "label":
            assert not np.isnan(args_dict["label"])
            assert args_dict["projection_distribution"] == "delta"
            assert np.isnan(args_dict["projection_index"])
            logger.warning(
                "projection_type is label, this means that the label will be used as the projection."
                "this is not a good idea for explainability best practices, because it will not be available at test time.",
            )
        elif args_dict["projection_type"] == "prediction":
            assert args_dict["projection_distribution"] is not None
            assert np.isnan(
                args_dict["projection_index"]
            ), "when projection is prediction, projection_index will be inferred from the forward function"
            if args_dict["projection_distribution"] == "delta":
                assert args_dict["projection_top_k"] > 0
            elif args_dict["projection_distribution"] == "uniform":
                assert args_dict["projection_top_k"] > 1
            elif args_dict["projection_distribution"] == "categorical":
                assert args_dict["projection_top_k"] > 1
            else:
                raise NotImplementedError

        elif args_dict["projection_type"] == "static":
            assert args_dict["projection_distribution"] == "delta"
            assert np.isnan(args_dict["projection_top_k"])
            assert args_dict["projection_index"] >= 0
        else:
            raise NotImplementedError

    @staticmethod
    def _process_logics_baseline_mask(args_dict):
        if args_dict["baseline_mask_type"] == "static":
            assert not np.isnan(args_dict["baseline_mask_value"])
        elif "gaussian" in args_dict["baseline_mask_type"]:
            assert np.isnan(args_dict["baseline_mask_value"])

    @staticmethod
    def _process_logics_alpha_mask(args_dict):
        if "static" == args_dict["alpha_mask_type"]:
            assert not np.isnan(args_dict["alpha_mask_value"])
        elif "uniform" in args_dict["alpha_mask_type"]:
            assert np.isnan(args_dict["alpha_mask_value"])

    @staticmethod
    def _process_combination_fn(args_dict):
        if args_dict["combination_fn"] == "convex":
            args_dict["combination_fn"] = operations.convex_combination
        elif args_dict["combination_fn"] == "additive":
            args_dict["combination_fn"] = operations.additive_combination
        elif args_dict["combination_fn"] == "damping":
            args_dict["combination_fn"] = operations.damping_combination
        else:
            raise NotImplementedError
        return args_dict

    @staticmethod
    def _process_explainer_fn(args_dict):
        if args_dict["explainer_fn"] == "vanilla_grad":
            args_dict["explainer_fn"] = explainers.vanilla_gradient
        elif args_dict["explainer_fn"] == "finite_difference":
            args_dict["explainer_fn"] = explainers.finite_difference
        else:
            raise NotImplementedError
        return args_dict

    @staticmethod
    def _process_alpha_mask(args_dict):
        if args_dict["alpha_mask_type"] == "static":
            alpha_mask = args_dict["alpha_mask_value"] * jnp.ones(shape=(1, 1, 1, 1))
        elif args_dict["alpha_mask_type"] == "scalar_uniform":  # stochastic IG
            alpha_mask = partial(
                jax.random.uniform,
                shape=(1, 1, 1, 1),
            )
        elif "image_bernoulli" in args_dict["alpha_mask_type"]:  # RISE
            assert not np.isnan(
                args_dict["alpha_mask_value"]
            ), "alpha_mask_value must be a valid probability for image_bernoulli"
            H_W_ = args_dict["alpha_mask_type"].split("-")[1]
            H, W = H_W_.split("x")
            H, W = int(H), int(W)
            I_H, I_W = args_dict["input_shape"][1], args_dict["input_shape"][2]
            assert (
                H <= I_H
            ), f"alpha_mask H {H} must be less than or equal to input_shape H {I_H}"
            assert (
                W <= I_W
            ), f"alpha_mask W {W} must be less than or equal to input_shape W {I_W}"
            C_H, C_W = I_H // H, I_W // W

            def alpha_mask_fn(key):
                alpha_mask = jax.random.bernoulli(
                    key=key,
                    p=args_dict["alpha_mask_value"],
                    shape=(1, H, W, 1),
                )
                # resize to input shape bilinear interpolation
                alpha_mask = jax.image.resize(
                    alpha_mask,
                    shape=(1, (H + 1) * C_H, (W + 1) * C_W, 1),
                    method="bilinear",
                )
                # sample random integer from 0 to C_H and C_W
                random_H = jax.random.randint(
                    key=key,
                    minval=0,
                    maxval=C_H,
                    shape=(1,),
                )
                random_W = jax.random.randint(
                    key=key,
                    minval=0,
                    maxval=C_W,
                    shape=(1,),
                )
                # crop to input shape
                alpha_mask = alpha_mask[
                    :,
                    random_H[0] : random_H[0] + I_H,
                    random_W[0] : random_W[0] + I_W,
                    :,
                ]

                return alpha_mask

            args_dict["alpha_mask"] = alpha_mask_fn
        else:
            raise NotImplementedError

        args_dict["alpha_mask"] = alpha_mask
        return args_dict

    @staticmethod
    def _process_baseline_mask(args_dict):
        if args_dict["baseline_mask_type"] == "static":
            baseline_mask = args_dict["baseline_mask_value"] * jnp.ones(
                shape=args_dict["input_shape"]
            )
        elif args_dict["baseline_mask_type"] == "gaussian":
            baseline_mask = partial(
                jax.random.normal,
                shape=args_dict["input_shape"],
            )
        elif "gaussian" in args_dict["baseline_mask_type"]:
            scale = args_dict["baseline_mask_type"].split("-")[1]
            scale = float(scale)
            baseline_mask = lambda key: scale * jax.random.normal(
                key=key,
                shape=args_dict["input_shape"],
            )
        else:
            raise NotImplementedError
        args_dict["baseline_mask"] = baseline_mask
        return args_dict

    @staticmethod
    def _process_projection(args_dict):
        if args_dict["projection_type"] == "label":
            temp_projection = operations.static_projection(
                num_classes=args_dict["num_classes"],
                index=args_dict["label"],
            )
            temp_projection_index = args_dict["label"]
        elif args_dict["projection_type"] == "prediction":
            if args_dict["projection_distribution"] == "categorical":
                (
                    temp_projection,
                    temp_projection_index,
                ) = operations.topk_categorical_random_projection(
                    image=args_dict["image"],
                    forward=args_dict["forward"],
                    k=args_dict["projection_top_k"],
                )
            elif args_dict["projection_distribution"] == "delta":
                """
                generates a delta distribution on the top k'th prediction.
                """
                (
                    temp_projection_index,
                    temp_projection,
                ) = operations.topk_static_projection(
                    image=args_dict["image"],
                    forward=args_dict["forward"],
                    k=args_dict["projection_top_k"],
                )
            elif args_dict["projection_distribution"] == "uniform":
                """
                generates a uniform distribution on top k predictions.
                """
                (
                    temp_projection_index,
                    temp_projection,
                ) = operations.topk_uniform_projection(
                    image=args_dict["image"],
                    forward=args_dict["forward"],
                    k=args_dict["projection_top_k"],
                )
            else:
                raise NotImplementedError
        elif args_dict["projection_type"] == "static":
            """
            generates a delta distribution on a single class specified by projection index.
            """
            temp_projection = operations.static_projection(
                num_classes=args_dict["num_classes"],
                index=args_dict["projection_index"],
            )
            temp_projection_index = args_dict["projection_index"]
        else:
            raise NotImplementedError

        args_dict["projection"] = temp_projection
        args_dict["projection_index"] = temp_projection_index
        return args_dict

    @classmethod
    def sample_demo(cls, static_kwargs, dynamic_kwargs, meta_kwargs):
        # we run a demo (one step of the algorithm after computations finished)
        static_kwargs = static_kwargs.copy()
        key = jax.random.PRNGKey(meta_kwargs["seed"])
        static_kwargs["demo"] = True
        static_kwargs["key"] = key
        static_kwargs.update(dynamic_kwargs)
        demo_output = cls._create_sampler(static_kwargs)()
        return demo_output
