from typing import Any, Callable, Optional, Dict, Tuple, Union
import jax
import jax.numpy as jnp

# from tqdm import tqdm
from source.operations import AbstractProcess


def forward_with_projection(inputs, projection, forward):
    assert inputs.ndim == 4, "x should be a batch of images"
    log_prob = forward(inputs)
    results_at_projection = (log_prob @ projection).squeeze()
    return results_at_projection, (results_at_projection, log_prob)


def vanilla_gradient(
    *,
    source_mask: jax.Array,
    projection: jax.Array,
    forward,
):
    assert len(source_mask.shape) == 4, "x should be a batch of images"
    grads, (results_at_projection, log_probs) = jax.grad(
        forward_with_projection,
        has_aux=True,
    )(source_mask, projection, forward)
    return grads, results_at_projection, log_probs


# class FiniteDifference(Explainer):
#     def __init__(
#         self,
#         model,
#     ) -> None:
#         """
#         computes the finite difference approximation of the gradient of the model
#         formally, D[f] = f(z) - f(x) where z ~ Neighborhood(x)
#         """
#         raise NotImplementedError("FiniteDifference is not implemented yet")
#         super().__init__(model)
#         self.y_hat_x: Union[jax.Array, None] = None

#     def __call__(
#         self,
#         input: jax.Array,
#         neighbor: jax.Array,
#         label: Optional[jax.Array],
#         observation_mask: jax.Array,
#         **kwargs: Any,
#     ) -> jax.Array:
#         assert (
#             input.shape == neighbor.shape
#         ), f"input shape and neighbor shape must match {input.shape} != {neighbor.shape}"
#         assert (
#             input.shape[2:] == observation_mask.shape[2:]
#         ), f"input and observation spatial shapes must match {input.shape} != {observation_mask.shape}"

#         x = input.clone().detach().requires_grad_(False)
#         if self.y_hat_x is None:
#             self.y_hat_x = self.model(x)
#             assert isinstance(self.y_hat_x, jax.Array)
#             if label is None:
#                 self.y = self.y_hat_x.argmax()
#             else:
#                 self.y = label
#             self.y_hat_x = self.y_hat_x.gather(1, self.y.view(-1, 1)).squeeze()
#         y_hat_z = self.model(neighbor)
#         assert isinstance(y_hat_z, jax.Array)

#         y_hat_z = y_hat_z.gather(1, self.y.view(-1, 1)).squeeze()
#         return (y_hat_z - self.y_hat_x) * observation_mask

#     def reset(self):
#         self.y_hat_x = None
