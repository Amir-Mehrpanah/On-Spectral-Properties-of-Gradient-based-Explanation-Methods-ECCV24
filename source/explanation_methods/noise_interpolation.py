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
    @staticmethod
    @AbstractFunction
    def sampler(
        key,
        projection,
        forward,
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

    def inplace_process_args(self, args):
        mixed_args = self.extract_mixed_args(args)
        mixed_pattern = self.extract_mixed_pattern(args.args_pattern, mixed_args)
        mixed_args = self.maybe_broad_cast_shapes(mixed_pattern, mixed_args)
        combined_patterns = combine_patterns(mixed_pattern, mixed_args)
        raise ValueError(combined_patterns[0])
        self._process_logics(combined_patterns)
        self.inplace_process_projection(args)
        self.inplace_process_baseline_mask(args)
        self.inplace_process_alpha_mask(args)
        self.inplace_process_sampler_args(args)
        self.inplace_create_sampler(args)

    @staticmethod
    def extract_mixed_pattern(args_pattern, mixed_args):
        # explicit pattern dependency:
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
        # forward <-- {}
        # demo <-- forward (fixed to False)

        args_pattern["label"] = args_pattern["projection"]

        mixed_pattern = {}
        for arg_name in mixed_args:
            pattern_proposal = [v for k, v in args_pattern.items() if k in arg_name]
            assert (
                len(pattern_proposal) == 1
            ), f"{arg_name} cannot be uniquely identified according to the provided {args_pattern}"
            mixed_pattern[arg_name] = pattern_proposal[0]
        return mixed_pattern

    def inplace_create_sampler(self, args):
        nones = tuple(None for _ in args.dynamic_args[0])
        samplers = [
            self.sampler(**compilation_pattern).concretize()
            for compilation_pattern in args.static_kwargs
        ]
        vectorized_samplers = [jax.vmap(v, in_axes=(0, *nones)) for v in samplers]
        args.samplers = vectorized_samplers

    def inplace_process_sampler_args(self, args):
        self.inplace_sort_dynamic_args(args)
        self.inplace_process_logics_sampler_args(args)
        combined_patterns = None
        args.dynamic_args, args.static_kwargs = self._split_args(
            combined_patterns,
            force_dynamic_args=args.dynamic_args,
        )

    def inplace_sort_dynamic_args(self, args):
        sampler_args_order = list(self.sampler.params.keys())
        sampler_args_order.remove("key")
        assert all(
            arg in args.args_pattern for arg in args.dynamic_args
        ), "dynamic_args contains an arg that is not in args_pattern please check the provided args"
        assert not "key" in args.args_pattern, "key is a reserved word"
        assert all(
            arg in sampler_args_order for arg in args.args_pattern
        ), "args_pattern contains an arg that is not in sampler args please check the provided args"
        assert all(
            arg in args.args_pattern for arg in sampler_args_order
        ), "sampler expects an argument that is not in args_pattern please check the provided args"
        # sort dynamic args according to sampler args order
        args.args_pattern = {arg: args.args_pattern[arg] for arg in sampler_args_order}

    @staticmethod
    def extract_mixed_args(args):
        mixed_args = {}
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
        ]
        for arg_name in input_args:
            assert hasattr(
                args, arg_name
            ), f"method expects an arg that is not in the provided args: {arg_name}"
            mixed_args[arg_name] = getattr(args, arg_name)

        mixed_args["demo"] = [False] * len(mixed_args["forward"])
        return mixed_args

    @staticmethod
    def _split_args(combined_patterns, force_dynamic_args):
        dynamic_args = []
        static_kwargs = []
        for combination in combined_patterns:
            dynamic_args.append({k: combination[k] for k in force_dynamic_args})
            static_kwargs.append(
                {k: combination[k] for k in combination if k not in force_dynamic_args}
            )
        return dynamic_args, static_kwargs

    def _process_logics(self, args):
        assert len(args.input_shape) == 4
        self._process_logics_alpha_mask(args)
        self._process_logics_baseline_mask(args)
        self._process_logics_projection(args)

    def inplace_broadcast_args(self, args):
        # broadcast alpha mask values according to alpha mask type
        args.alpha_mask_value = self._maybe_broadcast_arg(
            args.alpha_mask_value, args.alpha_mask_type
        )

        # broadcast label, image, projection distribution, forward, projection top k and projection index according to projection type
        args.label = self._maybe_broadcast_arg(args.label, args.projection_type)
        args.image = self._maybe_broadcast_arg(args.image, args.projection_type)
        args.forward = self._maybe_broadcast_arg(args.forward, args.projection_type)
        args.demo = self._maybe_broadcast_arg([False], args.projection_type)
        args.projection_distribution = self._maybe_broadcast_arg(
            args.projection_distribution, args.projection_type
        )
        args.projection_top_k = self._maybe_broadcast_arg(
            args.projection_top_k, args.projection_type
        )
        args.projection_index = self._maybe_broadcast_arg(
            args.projection_index, args.projection_type
        )

        # broadcast baseline mask values according to baseline mask type
        args.normalize_sample = self._maybe_broadcast_arg(
            args.normalize_sample, args.baseline_mask_type
        )
        args.baseline_mask_value = self._maybe_broadcast_arg(
            args.baseline_mask_value, args.baseline_mask_type
        )

    @staticmethod
    def maybe_broad_cast_shapes(pattern, values):
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

    def _process_logics_projection(self, args):
        iter_args = zip(
            args.projection_type,
            args.projection_distribution,
            args.projection_top_k,
            args.projection_index,
            args.label,
        )
        for (
            projection_type,
            projection_distribution,
            projection_top_k,
            projection_index,
            label,
        ) in iter_args:
            if projection_type == "label":
                assert label is not None
                assert projection_distribution is None
                assert projection_index is None
            elif projection_type == "random":
                assert projection_index is None
                assert projection_distribution is not None
                assert projection_top_k > 0
            elif projection_type == "prediction":
                assert projection_index is None
                assert projection_distribution is not None
                assert projection_top_k > 0
            elif projection_type == "static":
                assert projection_distribution is None
                assert projection_index is not None and projection_index >= 0
            elif projection_type == "ones":
                assert projection_distribution is None
                assert projection_index is None
                assert projection_top_k is None

    def _process_logics_baseline_mask(self, args):
        for baseline_mask_type, baseline_mask_value, normalize_sample in zip(
            args.baseline_mask_type,
            args.baseline_mask_value,
            args.normalize_sample,
        ):
            if baseline_mask_type == "static":
                assert baseline_mask_value is not None
                if isinstance(baseline_mask_value, float):
                    assert (
                        normalize_sample is False
                    ), "normalization of convex interpolation with static baseline is not expected"
            elif baseline_mask_type == "gaussian":
                assert baseline_mask_value is None

    def _process_logics_alpha_mask(self, args):
        for alpha_mask_type, alpha_mask_value in zip(
            args.alpha_mask_type, args.alpha_mask_value
        ):
            if alpha_mask_type == "static":
                assert alpha_mask_value is not None
            elif alpha_mask_type == "uniform":
                assert alpha_mask_value is None

    def inplace_process_alpha_mask(self, args):
        alpha_mask = []
        for alpha_mask_type, alpha_mask_value in zip(
            args.alpha_mask_type, args.alpha_mask_value
        ):
            if alpha_mask_type == "static":
                alpha_mask.append(alpha_mask_value * jnp.ones(shape=(1, 1, 1, 1)))
            elif alpha_mask_type == "scalar_uniform":
                alpha_mask.append(
                    partial(
                        jax.random.uniform,
                        shape=(1, 1, 1, 1),
                    )
                )
            elif alpha_mask_type == "image_uniform":
                alpha_mask.append(
                    partial(
                        jax.random.uniform,
                        shape=args.input_shape,
                    )
                )
            else:
                raise NotImplementedError

        args.alpha_mask = alpha_mask

    def inplace_process_baseline_mask(self, args):
        baseline_mask = []
        for baseline_mask_type, baseline_mask_value in zip(
            args.baseline_mask_type, args.baseline_mask_value
        ):
            if baseline_mask_type == "static":
                baseline_mask.append(
                    baseline_mask_value * jnp.ones(shape=args.input_shape)
                )
            elif baseline_mask_type == "gaussian":
                baseline_mask.append(
                    partial(
                        jax.random.normal,
                        shape=args.input_shape,
                    )
                )
            else:
                raise NotImplementedError

        args.baseline_mask = baseline_mask

    def inplace_process_projection(self, args):
        projection = []
        projection_indices = []

        iter_args = zip(
            args.projection_type,
            args.projection_distribution,
            args.projection_top_k,
            args.projection_index,
            args.label,
            args.image,
            args.forward,
        )
        for (
            projection_type,
            projection_distribution,
            projection_top_k,
            projection_index,
            label,
            image,
            forward,
        ) in iter_args:
            if projection_type == "label":
                temp_projection = operations.static_projection(
                    num_classes=args.num_classes,
                    index=label,
                )
                temp_projection_index = label
            elif projection_type == "random":
                temp_projection = operations.random_projection(
                    image=image,
                    forward=forward,
                    distribution=projection_distribution,
                    k=projection_top_k,
                )
                temp_projection_index = None
            elif projection_type == "prediction":
                if projection_distribution == "delta":
                    """
                    generates a delta distribution on the top k'th prediction.
                    """
                    (
                        temp_projection_index,
                        temp_projection,
                    ) = operations.prediction_projection(
                        image=image,
                        forward=forward,
                        k=projection_top_k,
                    )
                elif projection_distribution == "uniform":
                    """
                    generates a uniform distribution on top k predictions.
                    """
                    (
                        temp_projection_index,
                        temp_projection,
                    ) = operations.top_k_uniform_prediction_projection(
                        image=image,
                        forward=forward,
                        k=projection_top_k,
                    )
                elif projection_distribution == "categorical":
                    """
                    generates a categorical distribution on top k predictions.
                    the result is a list of one-hot vectors.
                    """
                    (
                        temp_projection_index,
                        temp_projection,
                    ) = operations.top_k_prediction_projection(
                        image=image,
                        forward=forward,
                        k=projection_top_k,
                    )
                else:
                    raise NotImplementedError

            elif projection_type == "ones":
                """
                generates a uniform distribution on all classes.
                """
                temp_projection = operations.ones_projection(
                    num_classes=args.num_classes,
                )
                temp_projection_index = None
            elif projection_type == "static":
                """
                generates a delta distribution on a single class specified by projection index.
                """
                temp_projection = operations.static_projection(
                    num_classes=args.num_classes,
                    index=projection_index,
                )
                temp_projection_index = projection_index
            else:
                raise NotImplementedError

            if isinstance(temp_projection, list):
                projection.extend(temp_projection)
                projection_indices.extend(temp_projection_index)
            else:
                projection.append(temp_projection)
                projection_indices.append(temp_projection_index)

        args.projection = projection
        args.projection_index = projection_indices

    @staticmethod
    def _maybe_broadcast_arg(arg, target_arg):
        if len(arg) == len(target_arg):
            return arg
        elif len(arg) == 1:
            return arg * len(target_arg)
        else:
            raise ValueError(
                "arg and target_arg must have the same length or arg must be of length 1"
            )

    def inplace_delete_extra_metadata_after_computation(
        self,
        args: argparse.Namespace,
    ):
        # things we added for computation but don't want to be saved as metadata
        del args.alpha_mask
        del args.projection
        del args.baseline_mask

    def inplace_pretty_print(self, pretty_kwargs: Dict):
        # things we added but don't want to be shown in the pretty print
        del pretty_kwargs["projection"]
        del pretty_kwargs["alpha_mask"]
        del pretty_kwargs["baseline_mask"]
        pretty_kwargs["projection_index"] = str(pretty_kwargs["projection_index"])

        return pretty_kwargs

    def inplace_demo(self, args):
        # we run a demo (one step of the algorithm after computations finished)
        key = jax.random.PRNGKey(args.seed)
        kwargs = self.inplace_process_args(args)
        demo_output = self.sampler(demo=True, **kwargs).concretize()(key=key)
        args.stats.update(demo_output)
