from typing import Any, Callable, Optional, Dict, Tuple, Union
import jax

from source.model_manager import forward_with_projection


def vanilla_gradient(
    *,
    forward,
    inputs,
    **kwargs,
):
    grads, aux = jax.grad(
        forward,
        has_aux=True,
    )(inputs)
    return grads, aux


def finite_difference(
    *,
    forward,
    inputs,
    alpha_mask,
    **kwargs,
):
    results_at_projection, log_prob = forward(inputs)
    return alpha_mask * results_at_projection, log_prob
