from typing import Any, Callable, Optional, Dict, Tuple, Union
from torch.func import functional_call, grad, vmap  # type: ignore
from functorch.experimental import replace_all_batch_norm_modules_
import torch

# from tqdm import tqdm
from source.utils import Statistic, rename_dict, Statistics
from source.neighborhoods import Distribution


class Explainer:
    def __init__(
        self,
        model: torch.nn.Module,
        projection: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        assert projection is None or (
            projection.ndim == 2 and projection.shape[1] == 1
        ), "projection is expected to be of shape (C,1)"
        self.projection: Optional[torch.Tensor] = projection

    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
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
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sample = {"input": x, "label": y}
            sample = self.distribution(sample)
            assert sample["input"].ndim == 3, "input should be of shape (C, H, W)"
            assert (
                sample["label"] is None or sample["label"].ndim == 1
            ), "label should be None or of shape (B,)"
            assert (
                sample["neighbor"].ndim == 4
            ), "neighbor should be of shape (N, C, H, W)"

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
        self.preds = Statistics(stats=prediction_stats)

    def __call__(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None, dim: int = 0
    ) -> Dict[str, torch.Tensor]:
        assert x.ndim == 3, "input tensor is expected to be of shape (C,H,W)"
        assert y is None or y.ndim == 1, "label tensor is expected to be of shape (N,)"

        for _ in range(self.num_batches):
            explanations, predictions = self.probabilistic_explainer(x, y)
            assert (
                explanations.ndim == 4
            ), "explanation tensor is expected to be of shape (N,C,H,W)"
            assert (
                predictions.ndim == 2
            ), "prediction tensor is expected to be of shape (N, 1)"

            with torch.no_grad():
                self.exps.update_statistics(explanations.detach().cpu())
                self.preds.update_statistics(predictions.detach().cpu())
        out_dict = self.exps.get_statistics()
        out_dict.update(self.preds.get_statistics())
        return out_dict


def stateless_forward_with_projection(
    sample: torch.Tensor,
    projection: torch.Tensor,
    model: torch.nn.Module,
    params: Tuple,
    returns: str = "log_probs",
):
    batch = sample.unsqueeze(0)
    predictions = functional_call(model, params, batch)
    log_probabilities = torch.log_softmax(predictions, dim=1)
    projected_log_probabilities = (log_probabilities @ projection).squeeze(1)
    if returns == "log_probs":
        aug_output = projected_log_probabilities
    else:
        aug_output = torch.exp(projected_log_probabilities)
    return projected_log_probabilities.squeeze(0), aug_output


class VanillaGradient(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        projection: Optional[torch.Tensor] = None,
        returns: str = "probs",
    ):
        """
        Args:
            model: a pytorch model
            projection: a projection vector of shape (C,1)
        This class implements the vanilla gradient explainer
        as computing the gradient of the output w.r.t the input
        is very expensive, we use the projection of the output
        to compute the gradient of the output @ projection w.r.t the input
        """

        # functional call does not pass tests due to high error introduced after vmap
        # essentially, the implementation takes same time as the naive implementation

        # must use replace_all_batch_norm_modules_ function to replace all batch norm modules
        # with their functional counterparts but it does not pass tests.
        super().__init__(model, projection)
        self.ft_compute_sample_grad = grad(
            stateless_forward_with_projection,
            has_aux=True,
        )
        assert returns in ["log_probs", "probs"], "returns should be log_probs or probs"
        self.returns = returns
        # self.ft_compute_batch_grad = vmap(
        #     ft_compute_sample_grad,
        #     in_dims=(
        #         0,
        #         None,
        #         None,
        #         None,
        #         None,
        #     ),
        # )
        self.detached_params = {k: v.detach() for k, v in self.model.named_parameters()}

    def __call__(
        self,
        neighbor: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert neighbor.ndim == 4, "neighbor is expected to be of shape (N,C,H,W)"

        if self.projection is None:
            output = self.model(neighbor[[0]])
            idx = output.argmax(dim=1)
            self.projection = torch.zeros_like(output.T)
            self.projection[idx, ...] = 1.0

        func_per_sample_grads = []
        func_preds = []
        for sample in neighbor:
            per_sample_grads, preds = self.ft_compute_sample_grad(
                sample,
                self.projection,
                self.model,
                self.detached_params,
                self.returns,
            )
            func_per_sample_grads.append(per_sample_grads)
            func_preds.append(preds)
        func_per_sample_grads = torch.stack(func_per_sample_grads)
        func_preds = torch.stack(func_preds)

        # func_per_sample_grads, func_preds = self.ft_compute_batch_grad(
        #     neighbor,
        #     self.projection,
        #     self.model,
        #     self.detatched_params,
        #     self.return_logprobs,
        # )

        return func_per_sample_grads, func_preds

    def reset(self):
        self.projection = None


class ScaledVanillaGradient(VanillaGradient):
    def __init__(
        self,
        model: torch.nn.Module,
        observation_mask: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
        projection: Optional[torch.Tensor] = None,
        returns: str = "probs",
    ):
        super().__init__(
            model,
            projection=projection,
            returns=returns,
        )
        self.observation_mask = observation_mask

    def __call__(
        self,
        neighbor: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad, preds = super().__call__(neighbor)
        if isinstance(self.observation_mask, torch.Tensor):
            return grad * self.observation_mask, preds
        return grad * self.observation_mask(neighbor), preds


class FiniteDifference(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        """
        computes the finite difference approximation of the gradient of the model
        formally, D[f] = f(z) - f(x) where z ~ Neighborhood(x)
        """
        super().__init__(model)
        self.y_hat_x: Union[torch.Tensor, None] = None

    def __call__(
        self,
        input: torch.Tensor,
        neighbor: torch.Tensor,
        label: Optional[torch.Tensor],
        observation_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert (
            input.shape == neighbor.shape
        ), f"input shape and neighbor shape must match {input.shape} != {neighbor.shape}"
        assert (
            input.shape[2:] == observation_mask.shape[2:]
        ), f"input and observation spatial shapes must match {input.shape} != {observation_mask.shape}"

        x = input.clone().detach().requires_grad_(False)
        if self.y_hat_x is None:
            self.y_hat_x = self.model(x)
            assert isinstance(self.y_hat_x, torch.Tensor)
            if label is None:
                self.y = self.y_hat_x.argmax()
            else:
                self.y = label
            self.y_hat_x = self.y_hat_x.gather(1, self.y.view(-1, 1)).squeeze()
        y_hat_z = self.model(neighbor)
        assert isinstance(y_hat_z, torch.Tensor)

        y_hat_z = y_hat_z.gather(1, self.y.view(-1, 1)).squeeze()
        return (y_hat_z - self.y_hat_x) * observation_mask

    def reset(self):
        self.y_hat_x = None
