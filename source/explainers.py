from typing import Any, Callable, Optional, Dict, Tuple, Union
import jax
import jax.numpy as jnp

# from tqdm import tqdm
from source.utils import Statistic, rename_dict, Statistics
from source.neighborhoods import Distribution


class Explainer:
    def __init__(
        self,
        forward: Callable,
    ) -> None:
        self.forward = forward

    def __call__(
        self,
        x: jax.Array,
        y: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        raise NotImplementedError("Explainer is an abstract class")

    def reset(self):
        raise NotImplementedError("Explainer is an abstract class")


class ProbabilisticExplainer:
    def __init__(
        self,
        deterministic_explainer: Explainer,
        distribution: Distribution,
        distribution_keys_to_explainer_args_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.distribution = distribution
        self.deterministic_explainer = deterministic_explainer
        self.keys_to_args_mapping = distribution_keys_to_explainer_args_mapping

    def __call__(
        self, x: jax.Array, y: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        sample = {"input": x, "label": y}
        sample = self.distribution(sample)
        assert sample["input"].ndim == 3, "input should be of shape (H, W, C)"
        assert (
            sample["label"] is None or sample["label"].ndim == 1
        ), "label should be None or of shape (B,)"
        assert sample["neighbor"].ndim == 4, "neighbor should be of shape (N, H, W, C)"

        if self.keys_to_args_mapping is not None:
            sample = rename_dict(sample, self.keys_to_args_mapping)

        explanation, prediction = self.deterministic_explainer(**sample)
        return explanation, prediction


class Aggregator:
    def __init__(
        self,
        probabilistic_explainer: ProbabilisticExplainer,
        explanation_stats: Dict[str, Statistic],
        prediction_stats: Dict[str, Statistic],
        batch_size: int,
    ) -> None:
        num_samples = list(explanation_stats.values())[0].num_samples
        assert num_samples == list(prediction_stats.values())[0].num_samples, (
            "number of samples for explanations and predictions should match, "
            f"got {num_samples} and {list(prediction_stats.values())[0].num_samples}"
        )
        assert (
            num_samples % batch_size == 0
        ), "number of samples should be divisible by batch size"
        self.num_batches = num_samples // batch_size
        self.probabilistic_explainer = probabilistic_explainer
        assert (
            set(explanation_stats.keys()) & set(prediction_stats.keys()) == set()
        ), "keys in explanation_stats and prediction_stats should not overlap"
        self.exps = Statistics(stats=explanation_stats)
        self.preds = Statistics(stats=prediction_stats)  # todo clean up

    def __call__(
        self, x: jax.Array, y: Optional[jax.Array] = None, dim: int = 0
    ) -> Dict[str, jax.Array]:
        assert x.ndim == 3, "input tensor is expected to be of shape (H,W,C)"
        assert y is None or y.ndim == 1, "label tensor is expected to be of shape (N,)"

        for _ in range(self.num_batches):
            explanations, predictions = self.probabilistic_explainer(x, y)
            assert (
                explanations.ndim == 4
            ), "explanation tensor is expected to be of shape (N,H,W,C)"
            assert (
                predictions.ndim == 2
            ), "prediction tensor is expected to be of shape (N, 1)"

            self.exps.update_statistics(explanations.detach().cpu())
            self.preds.update_statistics(predictions.detach().cpu())
        out_dict = self.exps.get_statistics()
        out_dict.update(self.preds.get_statistics())
        return out_dict


class VanillaGradient(Explainer):
    def __init__(
        self,
        forward: Callable,
    ):
        """
        Args:
            forward: the forward function of a jax flax model that returns the log probabilities of one class.
            It should only expect an input of shape (N,H,W,C) and return a tuple. The first element of the tuple
            should be a scalar and the second element should be log probabilities tensor of shape (N,1)

        This class implements the vanilla gradient explainer computes the gradient of the output w.r.t the input
        """
        super().__init__(forward)
        self.sample_grad_fn = jax.vmap(
            jax.grad(
                self.forward,
                argnums=0,
                has_aux=True,
            ),
            in_axes=(0),
        )

    def __call__(
        self,
        neighbor: jax.Array,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        assert neighbor.ndim == 4, "neighbor is expected to be of shape (N,H,W,C)"

        neighbor = jnp.expand_dims(neighbor, axis=1)
        per_sample_grads, log_probs = self.sample_grad_fn(neighbor)
        log_probs = log_probs.squeeze(1)
        per_sample_grads = per_sample_grads.squeeze(1)

        return per_sample_grads, log_probs


class ScaledVanillaGradient(VanillaGradient):
    def __init__(
        self,
        forward: Callable,
        observation_mask: Union[Callable[[jax.Array], jax.Array], jax.Array],
    ):
        super().__init__(forward)
        self.observation_mask = observation_mask

    def __call__(
        self,
        neighbor: jax.Array,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        grads, log_probs = super().__call__(neighbor)
        if isinstance(self.observation_mask, jax.Array):
            return grads * self.observation_mask, log_probs
        return grads * self.observation_mask(neighbor), log_probs


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
