import argparse
from functools import partial
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, List, Tuple
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from source.data_manager import minmax_normalize
from source.model_manager import forward_with_projection
from source import neighborhoods, explainers, operations
from source.utils import (
    Statistics,
    Stream,
    StreamNames,
    AbstractFunction,
    combine_patterns,
)


class TypeOrNone:
    def __init__(self, type) -> None:
        self.type = type

    def __call__(self, x) -> Any:
        if x == "None":
            return None
        return self.type(x)


class NoiseInterpolation:
    input_args = [
        "alpha_mask_type",
        "alpha_mask_value",
        "baseline_mask_type",
        "baseline_mask_value",
        "normalize_sample",
        "projection_type",
        "projection_distribution",
        "projection_top_k",
        "projection_index",
        "label",
        "image",
        "forward",
        "input_shape",
        "num_classes",
    ]

    @staticmethod
    @AbstractFunction
    def sampler(
        key,
        forward,
        projection,
        alpha_mask,
        image,
        baseline_mask,
        normalize_sample=True,
        demo=False,
    ):
        if isinstance(baseline_mask, Callable):
            baseline_mask = baseline_mask(key=key)
        if isinstance(projection, Callable):
            projection = projection(key=key)
        if isinstance(alpha_mask, Callable):
            alpha_mask = alpha_mask(key=key)

        convex_combination_mask = operations.convex_combination_mask(
            source_mask=image,
            target_mask=baseline_mask,
            alpha_mask=alpha_mask,
        )
        # set image to the expected range [0,1]
        # this transformation makes the resulting image
        # invariant to fixed perturbations of the image
        # (e.g. adding a constant to all pixels)
        if normalize_sample:
            convex_combination_mask = minmax_normalize(convex_combination_mask)
        (
            vanilla_grad_mask,
            results_at_projection,
            log_probs,
        ) = explainers.vanilla_gradient(
            forward=forward_with_projection,
            inputs=(convex_combination_mask, projection, forward),
        )

        if demo:
            return {
                Stream(
                    StreamNames.vanilla_grad_mask, Statistics.none
                ): vanilla_grad_mask,
                Stream(
                    StreamNames.results_at_projection, Statistics.none
                ): results_at_projection,
                Stream(StreamNames.log_probs, Statistics.none): log_probs,
                Stream(StreamNames.image, Statistics.none): image,
                Stream(
                    "convex_combination_mask", Statistics.none
                ): convex_combination_mask,
                Stream("projection", Statistics.none): projection,
                Stream("alpha_mask", Statistics.none): alpha_mask,
                Stream("baseline_mask", Statistics.none): baseline_mask,
            }
        return {
            StreamNames.vanilla_grad_mask: vanilla_grad_mask,
            StreamNames.results_at_projection: results_at_projection,
            StreamNames.log_probs: log_probs,
        }

    def inplace_add_args(self, base_parser):
        base_parser.add_argument(
            "--alpha_mask_type",
            type=str,
            required=True,
            nargs="+",
            choices=["static", "scalar_uniform"],
        )
        base_parser.add_argument(
            "--alpha_mask_value",
            type=TypeOrNone(type=float),
            nargs="*",
            default=[None],
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
            type=TypeOrNone(int),
            nargs="*",
            default=[None],
        )
        base_parser.add_argument(
            "--projection_index",
            type=TypeOrNone(int),
            nargs="*",
            default=[None],
        )
        base_parser.add_argument(
            "--projection_distribution",
            type=TypeOrNone(str),
            choices=[None, "uniform", "categorical", "delta"],
            default=["delta"],
            nargs="*",
        )
        base_parser.add_argument(
            "--baseline_mask_type",
            type=str,
            required=True,
            choices=["static", "gaussian"],
            nargs="+",
        )
        base_parser.add_argument(
            "--baseline_mask_value",
            type=TypeOrNone(float),
            default=[None],
            nargs="*",
        )
        base_parser.add_argument(
            "--normalize_sample",
            type=bool,
            default=[True],
            nargs="+",
        )

    @classmethod
    def inplace_process_args(cls, args):
        mixed_args = cls.extract_mixed_args(args)
        mixed_pattern = cls.extract_mixed_pattern(args.args_pattern, mixed_args)
        mixed_args = cls.maybe_broadcast_shapes(mixed_pattern, mixed_args)
        combined_mixed_args = combine_patterns(mixed_pattern, mixed_args)
        list(map(cls._process_logics, combined_mixed_args))
        combined_mixed_args = list(map(cls._process_args, combined_mixed_args))
        (
            combined_dynamic_kwargs,
            combined_static_kwargs,
            meta_kwargs,
        ) = cls._split_args_dicts(
            combined_mixed_args,
            args_state=args.args_state,
        )
        combined_dynamic_kwargs = list(
            map(
                cls._sort_dynamic_kwargs,
                combined_dynamic_kwargs,
            )
        )
        vmap_axis_for_one = (0,) + tuple(
            None for _ in combined_dynamic_kwargs[0]
        )  # 0 for key, None for dynamic args
        vmap_axis_for_all = [vmap_axis_for_one] * len(combined_dynamic_kwargs)
        samplers = list(
            map(
                cls._create_sampler,
                combined_static_kwargs,
                vmap_axis_for_all,
            )
        )

        args.samplers = samplers
        args.dynamic_kwargs = combined_dynamic_kwargs
        args.meta_kwargs = meta_kwargs
        args.static_kwargs = combined_static_kwargs

    @classmethod
    def _process_args(cls, args_dict):
        args_dict = cls._process_projection(args_dict)
        args_dict = cls._process_baseline_mask(args_dict)
        args_dict = cls._process_alpha_mask(args_dict)

        cls.sampler_args = list(cls.sampler.params.keys())
        assert cls.sampler_args[0] == "key", "key must be the first arg of sampler"
        cls.sampler_args.remove("key")
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
                    len_factor = max_value / len_values[i]
                    pattern_key = pattern_keys[i]
                    values[pattern_key] = np.repeat(values[pattern_key], len_factor)

        return values

    @classmethod
    def extract_mixed_pattern(cls, args_pattern, mixed_args):
        # explicit pattern dependency: todo make it more formal
        # A <-- B means that B will be inferred from A
        # projection <-- label
        # projection <-- projection_type
        # projection <-- projection_distribution
        # projection <-- projection_top_k
        # projection <-- projection_index
        # alpha_mask <-- alpha_mask_value
        # alpha_mask <-- alpha_mask_type
        # baseline_mask <-- baseline_mask_value
        # baseline_mask <-- baseline_mask_type
        # normalize_sample <-- {}
        # image <-- {}
        # image <-- input_shape
        # forward <-- {}
        # num_classes <-- forward
        # demo <-- forward (fixed to False)

        args_pattern["label"] = args_pattern["projection"]
        args_pattern["input_shape"] = args_pattern["image"]
        args_pattern["num_classes"] = args_pattern["forward"]

        mixed_pattern = {}
        for arg_name in mixed_args:
            pattern_proposal = [v for k, v in args_pattern.items() if k in arg_name]
            assert (
                len(pattern_proposal) == 1
            ), f"{arg_name} cannot be uniquely identified according to the provided {args_pattern}"
            mixed_pattern[arg_name] = pattern_proposal[0]
        return mixed_pattern

    @classmethod
    def _create_sampler(cls, static_kwargs, vamp_axis=None):
        sampler = cls.sampler(**static_kwargs).concretize()
        if vamp_axis is not None:
            sampler = jax.vmap(sampler, in_axes=vamp_axis)
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

    @classmethod
    def extract_mixed_args(cls, args):
        mixed_args = {}
        for arg_name in cls.input_args:
            assert hasattr(
                args, arg_name
            ), f"method expects an arg that is not in the provided args: {arg_name}"
            mixed_args[arg_name] = getattr(args, arg_name)

        mixed_args["demo"] = [False] * len(mixed_args["forward"])
        mixed_args["input_shape"] = [mixed_args["input_shape"]] * len(
            mixed_args["image"]
        )
        mixed_args["num_classes"] = [mixed_args["num_classes"]] * len(
            mixed_args["forward"]
        )
        return mixed_args

    @classmethod
    def _split_args_dicts(cls, combined_mixed_args, args_state):
        static_keys = [k for k, v in args_state.items() if "static" in v]
        dynamic_keys = [k for k, v in args_state.items() if "dynamic" in v]
        meta_keys = [k for k, v in args_state.items() if "meta" in v]
        assert len(static_keys) + len(dynamic_keys) == len(
            cls.sampler_args
        ), f"static and dynamic keys must be equal to sampler args {cls.sampler_args}"
        dynamic_kwargs = []
        static_kwargs = []
        meta_kwargs = []
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
            dynamic_kwargs.append(temp_dynamic_kwargs)
            static_kwargs.append(temp_static_kwargs)
            meta_kwargs.append(temp_meta_args)

        return dynamic_kwargs, static_kwargs, meta_kwargs

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

    @staticmethod
    def _process_logics_projection(args_dict):
        if args_dict["projection_type"] == "label":
            assert args_dict["label"] is not None
            assert args_dict["projection_distribution"] is None
            assert args_dict["projection_index"] is None
            print(
                "Warning: projection_type is label, this means that the label will be used as the projection."
                "this is not a good idea for explainability best practices, because it will not be available at test time."
            )
        elif args_dict["projection_type"] == "prediction":
            assert args_dict["projection_distribution"] is not None
            assert (
                args_dict["projection_index"] is None
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
            assert args_dict["projection_top_k"] is None
            assert args_dict["projection_index"] >= 0
        else:
            raise NotImplementedError

    @staticmethod
    def _process_logics_baseline_mask(args_dict):
        if args_dict["baseline_mask_type"] == "static":
            assert args_dict["baseline_mask_value"] is not None
            if isinstance(args_dict["baseline_mask_value"], float):
                assert (
                    args_dict["normalize_sample"] is False
                ), "normalization of convex interpolation with static baseline is not expected"
        elif args_dict["baseline_mask_type"] == "gaussian":
            assert args_dict["baseline_mask_value"] is None

    @staticmethod
    def _process_logics_alpha_mask(args_dict):
        if args_dict["alpha_mask_type"] == "static":
            assert args_dict["alpha_mask_value"] is not None
        elif args_dict["alpha_mask_type"] == "uniform":
            assert args_dict["alpha_mask_value"] is None

    @staticmethod
    def _process_alpha_mask(args_dict):
        if args_dict["alpha_mask_type"] == "static":
            alpha_mask = args_dict["alpha_mask_value"] * jnp.ones(shape=(1, 1, 1, 1))
        elif args_dict["alpha_mask_type"] == "scalar_uniform":
            alpha_mask = partial(
                jax.random.uniform,
                shape=(1, 1, 1, 1),
            )
        elif args_dict["alpha_mask_type"] == "image_uniform":
            alpha_mask = partial(
                jax.random.uniform,
                shape=args_dict["input_shape"],
            )
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
    def inplace_clean_junk_after_processing_args(
        cls,
        args: argparse.Namespace,
    ):
        # things we added for computation but don't need anymore
        for k in cls.input_args:
            delattr(args, k)
        del args.stats_log_level
        del args.output_layer
        del args.args_pattern
        del args.args_state

    @staticmethod
    def inplace_make_pretty(pretty_kwargs: List):
        for item in pretty_kwargs:
            item["projection_index"] = int(item["projection_index"])

    @classmethod
    def inplace_demo(cls, args, stats):
        # we run a demo (one step of the algorithm after computations finished)
        key = jax.random.PRNGKey(args.seed)
        kwargs = args.dynamic_kwargs[0]
        kwargs.update(args.static_kwargs[0])
        kwargs["demo"] = True
        kwargs["key"] = key
        demo_output = cls._create_sampler(kwargs)()
        stats.update(demo_output)
