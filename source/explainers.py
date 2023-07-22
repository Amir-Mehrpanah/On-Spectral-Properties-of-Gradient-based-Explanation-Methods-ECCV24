from typing import Any, Callable, Optional, Dict, Tuple, Union
import jax
import jax.numpy as jnp

# from tqdm import tqdm
from source.operations import Statistic, rename_dict, Statistics
from source.neighborhoods import Distribution


def make_vanilla_gradient(
    name: str,
    source_name: str,
    forward: Callable,
    has_aux: bool = False,
) -> Callable:
    """
    Args:
        name: name of the mask to be put in the stream
        source_name: name of the mask in the stream to be fed to the forward function
        forward: the forward function of a jax flax model that returns the log probabilities of one class.
        It should only expect an input of shape (N,H,W,C) and return a tuple. The first element of the tuple
        should be a scalar and the second element should be log probabilities tensor of shape (N,1)
        has_aux: whether the forward function returns a tuple of (output, aux) or just the output

    This class implements the vanilla gradient explainer computes the gradient of the output w.r.t the source_name
    """

    def vanilla_gradient(
        *,
        stream: jax.Array,
        key: jax.random.KeyArray,
    ) -> Tuple[jax.Array, jax.Array]:
        out = jax.grad(
            forward,
            argnums=0,
            has_aux=has_aux,
        )(stream[source_name])
        return out

    return vanilla_gradient


class FiniteDifference(Explainer):
    def __init__(
        self,
        model,
    ) -> None:
        """
        computes the finite difference approximation of the gradient of the model
        formally, D[f] = f(z) - f(x) where z ~ Neighborhood(x)
        """
        raise NotImplementedError("FiniteDifference is not implemented yet")
        super().__init__(model)
        self.y_hat_x: Union[jax.Array, None] = None

    def __call__(
        self,
        input: jax.Array,
        neighbor: jax.Array,
        label: Optional[jax.Array],
        observation_mask: jax.Array,
        **kwargs: Any,
    ) -> jax.Array:
        assert (
            input.shape == neighbor.shape
        ), f"input shape and neighbor shape must match {input.shape} != {neighbor.shape}"
        assert (
            input.shape[2:] == observation_mask.shape[2:]
        ), f"input and observation spatial shapes must match {input.shape} != {observation_mask.shape}"

        x = input.clone().detach().requires_grad_(False)
        if self.y_hat_x is None:
            self.y_hat_x = self.model(x)
            assert isinstance(self.y_hat_x, jax.Array)
            if label is None:
                self.y = self.y_hat_x.argmax()
            else:
                self.y = label
            self.y_hat_x = self.y_hat_x.gather(1, self.y.view(-1, 1)).squeeze()
        y_hat_z = self.model(neighbor)
        assert isinstance(y_hat_z, jax.Array)

        y_hat_z = y_hat_z.gather(1, self.y.view(-1, 1)).squeeze()
        return (y_hat_z - self.y_hat_x) * observation_mask

    def reset(self):
        self.y_hat_x = None
