from typing import Callable
import os
import jax
import sys

sys.path.append(os.getcwd())
from source.explanation_methods.noise_interpolation import NoiseInterpolation
from source.data_manager import minmax_normalize
from source.model_manager import forward_with_projection
from source import neighborhoods, explainers, operations
from source.utils import Statistics, Stream, StreamNames, AbstractFunction


class FisherInformation(NoiseInterpolation):
    @AbstractFunction
    def sample(
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
        if isinstance(alpha_mask, Callable):
            alpha_mask = alpha_mask(key=key)

        convex_combination_mask = operations.convex_combination_mask(
            source_mask=image,
            target_mask=baseline_mask,
            alpha_mask=alpha_mask,
        )
        convex_combination_mask = minmax_normalize(convex_combination_mask)
        sum_vanilla_grad_mask = 0
        for p_ in projection:
            (
                vanilla_grad_mask,
                results_at_projection,
                log_probs,
            ) = explainers.vanilla_gradient(
                forward=forward_with_projection,
                inputs=(convex_combination_mask, p_, forward),
            )
            sum_vanilla_grad_mask += vanilla_grad_mask

        if demo:
            raise NotImplementedError("demo is not implemented for FisherInformation")
        return {
            StreamNames.vanilla_grad_mask: sum_vanilla_grad_mask,
            StreamNames.results_at_projection: results_at_projection,
            StreamNames.log_probs: log_probs,
        }

    def inplace_process_logics(args):
        assert len(args.input_shape) == 4

        if args.alpha_mask_type == "static":
            assert args.alpha_mask_value is not None
        elif args.alpha_mask_type == "uniform":
            assert args.alpha_mask_value is None

        if args.baseline_mask_type == "static":
            assert args.baseline_mask_value is not None
        elif args.baseline_mask_type == "gaussian":
            assert args.baseline_mask_value is None

        if args.projection_type == "prediction":
            assert args.projection_distribution == "uniform"
            assert args.projection_top_k is not None and args.projection_top_k > 0
        else:
            raise NotImplementedError("other projection types are not implemented")

    def process_args(args):
        FisherInformation.inplace_process_logics(args)
        FisherInformation.inplace_process_projection(args)
        NoiseInterpolation.inplace_process_baseline(args)
        NoiseInterpolation.inplace_process_alpha_mask(args)

        return {
            "forward": args.forward,
            "alpha_mask": args.alpha_mask,
            "projection": args.projection,
            "image": args.image,
            "baseline_mask": args.baseline_mask,
        }

    def inplace_process_projection(args):
        if args.projection_type == "prediction":
            args.projection = operations.top_k_prediction_projection(
                image=args.image,
                forward=args.forward,
                k=args.projection_top_k,
            )
        else:
            raise NotImplementedError("other projection types are not implemented")


# class IntegratedGradient(Aggregator):
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         input_shape: Tuple[int, int, int],
#         prediction_stats: Dict[str, Statistic],
#         explanation_stats: Dict[str, Statistic],
#         baseline: torch.Tensor = torch.zeros(1, 1, 1, 1),
#         batch_size: int = 32,
#     ) -> None:
#         assert baseline.ndim == 4, "baseline is expected to be of shape (N,C,H,W)"
#         device = get_device(model)
#         mask_shape = (batch_size, 1, 1, 1)
#         baseline_dist = DeterministicMask(name="baseline", mask=baseline, device=device)
#         maybe_resize_baseline: List[Distribution] = []
#         alpha_name = "alpha"
#         if baseline.shape[-2:] != input_shape[-2:] and baseline.shape[-2:] != (1, 1):
#             alpha_name = "resized_" + alpha_name
#             maybe_resize_baseline.append(
#                 ResizeMask(
#                     name=alpha_name,
#                     source_name="alpha",
#                     target_name="input",
#                     device=device,
#                 )
#             )

#         distribution = Compose(
#             [
#                 UniformMask(name="alpha", shape=mask_shape, device=device),
#                 baseline_dist,
#                 *maybe_resize_baseline,
#                 ConvexCombination(
#                     device=device,
#                     name="neighbor",
#                     source_mask="baseline",
#                     alpha_mask=alpha_name,
#                     target_mask="input",
#                 ),
#             ]
#         )
#         scale: Callable[[torch.Tensor], torch.Tensor] = lambda x: x - baseline_dist.mask
#         deterministic_explainer = ScaledVanillaGradient(
#             model, observation_mask=scale, returns="probs"
#         )
#         probabilistic_explainer = ProbabilisticExplainer(
#             deterministic_explainer, distribution
#         )
#         super().__init__(
#             probabilistic_explainer,
#             prediction_stats=prediction_stats,
#             explanation_stats=explanation_stats,
#             batch_size=batch_size,
#         )


# class OcclusionSensitivity(Aggregator):
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         input_shape: Tuple[int, int, int],
#         explanation_stats: Dict[str, Statistic],
#         prediction_stats: Dict[str, Statistic],
#         baseline: torch.Tensor = torch.zeros(1, 1, 8, 8),
#         batch_size: int = 32,
#     ) -> None:
#         assert baseline.ndim == 4, "baseline is expected to be of shape (N,C,H,W)"
#         device = get_device(model)
#         for p in model.parameters():
#             p.requires_grad = False

#         maybe_resize_masks: List[Distribution] = []
#         alpha_name = "alpha"
#         baseline_name = "baseline"
#         if baseline.shape[-2:] != input_shape[-2:] and baseline.shape[-2:] != (1, 1):
#             alpha_name = "resized_" + alpha_name
#             baseline_name = "resized_" + baseline_name
#             maybe_resize_masks.append(
#                 ResizeMask(
#                     name=alpha_name,
#                     source_name="alpha",
#                     target_name="input",
#                     device=device,
#                 )
#             )
#             maybe_resize_masks.append(
#                 ResizeMask(
#                     name=baseline_name,
#                     source_name="baseline",
#                     target_name="input",
#                     device=device,
#                 ),
#             )

#         distribution = Compose(
#             [
#                 OneHotCategoricalMask(
#                     name="alpha", shape=(batch_size, *baseline.shape[1:]), device=device
#                 ),
#                 DeterministicMask(name="baseline", mask=baseline, device=device),
#                 *maybe_resize_masks,
#                 ConvexCombination(
#                     source_mask="input",
#                     alpha_mask=alpha_name,
#                     target_mask=baseline_name,
#                     name="neighbor",
#                     device=device,
#                 ),
#             ]
#         )
#         self.deterministic_explainer = FiniteDifference(model)
#         probabilistic_explainer = ProbabilisticExplainer(
#             self.deterministic_explainer,
#             distribution,
#             distribution_keys_to_explainer_args_mapping={
#                 alpha_name: "observation_mask",
#             },
#         )
#         super().__init__(
#             probabilistic_explainer,
#             explanation_stats=explanation_stats,
#             prediction_stats=prediction_stats,
#             batch_size=batch_size,
#         )

#     def __call__(self, x: torch.Tensor, dim: int = 0):
#         self.deterministic_explainer.reset()
#         return super().__call__(x, dim=dim)


# class RISE(Aggregator):
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         input_shape: Tuple[int, int, int],
#         baseline: torch.Tensor,
#         prediction_stats: Dict[str, Statistic],
#         explanation_stats: Dict[str, Statistic],
#         p_keep: float = 0.1,
#         batch_size: int = 32,
#     ) -> None:
#         assert baseline.ndim == 4, "baseline is expected to be of shape (N,C,H,W)"
#         device = get_device(model)
#         for p in model.parameters():
#             p.requires_grad = False

#         maybe_resize_masks: List[Distribution] = []
#         alpha_name = "alpha"
#         baseline_name = "baseline"
#         if baseline.shape != input_shape and baseline.shape[-2:] != (1, 1):
#             alpha_name = "resized_" + alpha_name
#             baseline_name = "resized_" + baseline_name
#             mask_target_shape = (
#                 int(input_shape[-2] * (1 + 1 / baseline.shape[-2]) + 1),
#                 int(input_shape[-1] * (1 + 1 / baseline.shape[-1]) + 1),
#             )
#             maybe_resize_masks.append(
#                 ResizeMask(
#                     name=alpha_name,
#                     source_name="alpha",
#                     # according to the official implementation of RISE, which is different from the paper,
#                     # the alpha mask is resized to the a size a bit bigger than the input shape and then
#                     # cropped to the input shape. this helps removing the border artifacts.
#                     target_shape=mask_target_shape,
#                     device=device,
#                 )
#             )
#             maybe_resize_masks.append(
#                 RandomCropMask(
#                     name=alpha_name,
#                     source_name=alpha_name,
#                     target_name="input",
#                     device=device,
#                 )
#             )
#             maybe_resize_masks.append(
#                 ResizeMask(
#                     name=baseline_name,
#                     source_name="baseline",
#                     target_name="input",
#                     device=device,
#                 ),
#             )

#         distribution = Compose(
#             [
#                 BernoulliMask(
#                     name="alpha",
#                     shape=(batch_size, *baseline.shape[1:]),
#                     device=device,
#                     p=p_keep,
#                 ),
#                 DeterministicMask(name="baseline", mask=baseline, device=device),
#                 *maybe_resize_masks,
#                 ConvexCombination(
#                     source_mask=baseline_name,
#                     alpha_mask=alpha_name,
#                     target_mask="input",
#                     name="neighbor",
#                     device=device,
#                 ),
#             ]
#         )
#         # for seeing how it could be finite difference see our paper
#         deterministic_explainer = FiniteDifference(model)
#         probabilistic_explainer = ProbabilisticExplainer(
#             deterministic_explainer,
#             distribution,
#             distribution_keys_to_explainer_args_mapping={
#                 alpha_name: "observation_mask",
#             },
#         )
#         super().__init__(
#             probabilistic_explainer,
#             prediction_stats=prediction_stats,
#             explanation_stats=explanation_stats,
#             batch_size=batch_size,
#         )
