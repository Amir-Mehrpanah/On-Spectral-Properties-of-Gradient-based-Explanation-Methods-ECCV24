import jax.numpy as jnp
from typing import Callable, Dict, List, Tuple
import os
import sys

sys.path.append(os.getcwd())
from source import neighborhoods, explainers, operations
from source.utils import StreamNames, AbstractFunction


@AbstractFunction
def noise_interpolation(key, *, alpha, forward, num_classes, input_shape, image, label):
    assert len(input_shape) == 4

    alpha_mask = alpha * jnp.ones(shape=(1, 1, 1, 1))
    projection = (
        jnp.zeros(
            shape=(num_classes, 1),
            dtype=jnp.float32,
        )
        .at[label, 0]
        .set(1.0)
    )

    normal_mask = neighborhoods.normal_mask(
        key=key,
        shape=input_shape,
    )

    convex_combination_mask = operations.convex_combination_mask(
        source_mask=image,
        target_mask=normal_mask,
        alpha_mask=alpha_mask,
    )

    vanilla_grad_mask, results_at_projection, log_probs = explainers.vanilla_gradient(
        source_mask=convex_combination_mask,
        projection=projection,
        forward=forward,
    )

    return {
        StreamNames.vanilla_grad_mask: vanilla_grad_mask,
        StreamNames.results_at_projection: results_at_projection,
        StreamNames.log_probs: log_probs,
    }


def inplace_noise_interpolation_parser(base_parser):
    base_parser.add_argument(
        "--alpha",
        type=float,
        required=True,
    )


def noise_interpolation_select_args(args):
    return {
        "alpha": args.alpha,
        "forward": args.forward,
        "num_classes": args.num_classes,
        "input_shape": args.input_shape,
        "image": args.image,
        "label": args.label,
    }
