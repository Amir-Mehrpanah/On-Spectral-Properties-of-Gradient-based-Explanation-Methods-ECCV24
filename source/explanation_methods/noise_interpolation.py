from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable, Dict, List, Tuple
import os
import sys

sys.path.append(os.getcwd())
from source.data_manager import minmax_normalize
from source.model_manager import forward_with_projection
from source import neighborhoods, explainers, operations
from source.utils import Statistics, Stream, StreamNames, AbstractFunction


@AbstractFunction
def noise_interpolation(
    key,
    *,
    forward,
    alpha_mask,
    projection,
    image,
    baseline_mask,
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
    convex_combination_mask = minmax_normalize(
        convex_combination_mask
    )  # set image to the correct range [0,1]
    vanilla_grad_mask, results_at_projection, log_probs = explainers.vanilla_gradient(
        forward=forward_with_projection,
        inputs=(convex_combination_mask, projection, forward),
    )

    if demo:
        return {
            Stream(StreamNames.vanilla_grad_mask, Statistics.none): vanilla_grad_mask,
            Stream(
                StreamNames.results_at_projection, Statistics.none
            ): results_at_projection,
            Stream(StreamNames.log_probs, Statistics.none): log_probs,
            Stream(StreamNames.image, Statistics.none): image,
            Stream("convex_combination_mask", Statistics.none): convex_combination_mask,
            Stream("projection", Statistics.none): projection,
            Stream("alpha_mask", Statistics.none): alpha_mask,
            Stream("baseline_mask", Statistics.none): baseline_mask,
        }
    return {
        StreamNames.vanilla_grad_mask: vanilla_grad_mask,
        StreamNames.results_at_projection: results_at_projection,
        StreamNames.log_probs: log_probs,
    }


def inplace_noise_interpolation_parser(base_parser):
    base_parser.add_argument(
        "--alpha_mask_type",
        type=str,
        required=True,
        choices=["static", "scalar_uniform"],
    )
    base_parser.add_argument(
        "--alpha_mask_value",
        type=float,
    )
    base_parser.add_argument(
        "--projection_type",
        type=str,
        required=True,
        choices=["label", "random", "prediction", "static"],
    )
    base_parser.add_argument(
        "--projection_top_k",
        type=int,
    )
    base_parser.add_argument(
        "--projection_index",
        type=int,
    )
    base_parser.add_argument(
        "--projection_distribution",
        type=str,
        choices=["uniform", "categorical"],
    )
    base_parser.add_argument(
        "--baseline_mask_type",
        type=str,
        required=True,
        choices=["static", "gaussian"],
    )
    base_parser.add_argument(
        "--baseline_mask_value",
        type=float,
    )


def noise_interpolation_process_args(args):
    inplace_process_logical(args)
    inplace_process_projection(args)
    inplace_process_baseline(args)
    inplace_process_alpha_mask(args)

    return {
        "forward": args.forward,
        "alpha_mask": args.alpha_mask,
        "projection": args.projection,
        "image": args.image,
        "baseline_mask": args.baseline_mask,
    }


def inplace_process_logical(args):
    assert len(args.input_shape) == 4

    if args.alpha_mask_type == "static":
        assert args.alpha_mask_value is not None
    elif args.alpha_mask_type == "uniform":
        assert args.alpha_mask_value is None

    if args.baseline_mask_type == "static":
        assert args.baseline_mask_value is not None
    elif args.baseline_mask_type == "gaussian":
        assert args.baseline_mask_value is None

    if args.projection_type == "label":
        assert args.label is not None
        assert args.projection_distribution is None
    elif args.projection_type == "random":
        assert args.projection_distribution is not None
    elif args.projection_type == "prediction":
        assert args.projection_distribution is None
        assert args.projection_top_k is not None
        assert args.projection_top_k > 0
    elif args.projection_type == "static":
        assert args.projection_index is not None
        assert args.projection_index >= 0


def inplace_process_alpha_mask(args):
    if args.alpha_mask_type == "static":
        args.alpha_mask = args.alpha_mask_value * jnp.ones(shape=(1, 1, 1, 1))
    elif args.alpha_mask_type == "scalar_uniform":
        args.alpha_mask = partial(
            jax.random.uniform,
            shape=(1, 1, 1, 1),
        )


def inplace_process_baseline(args):
    if args.baseline_mask_type == "static":
        args.baseline_mask = args.baseline_mask_value * jnp.ones(shape=args.input_shape)
    elif args.baseline_mask_type == "gaussian":
        args.baseline_mask = partial(
            jax.random.normal,
            shape=args.input_shape,
        )


def inplace_process_projection(args):
    if args.projection_type == "label":
        args.projection = operations.static_projection(
            num_classes=args.num_classes,
            index=args.label,
        )
    elif args.projection_type == "random":
        raise NotImplementedError
        args.projection = operations.random_projection(
            num_classes=args.num_classes,
            distribution=args.projection_distribution,
        )
    elif args.projection_type == "prediction":
        args.projection = operations.prediction_projection(
            image=args.image,
            forward=args.forward,
            k=args.projection_top_k,
        )
    elif args.projection_type == "static":
        args.projection = operations.static_projection(
            num_classes=args.num_classes,
            index=args.projection_index,
        )


def inplace_delete_noise_interpolation_extra_metadata(args):
    # things we added but don't want to be saved as metadata
    del args.alpha_mask
    del args.projection
    del args.baseline_mask


def inplace_noise_interpolation_pretty_print(pretty_kwargs):
    # things we added but don't want to be shown in the pretty print
    del pretty_kwargs["projection"]
    del pretty_kwargs["alpha_mask"]
    del pretty_kwargs["baseline_mask"]
    return pretty_kwargs


def inplace_noise_interpolation_demo(args):
    key = jax.random.PRNGKey(args.seed)
    kwargs = noise_interpolation_process_args(args)
    demo_output = noise_interpolation(demo=True, **kwargs).concretize()(key=key)
    args.stats.update(demo_output)
