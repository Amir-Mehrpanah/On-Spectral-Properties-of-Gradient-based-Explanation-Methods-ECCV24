from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms

import source.utils as utils
from source.utils import SmartDevice


class Distribution(SmartDevice):
    def __init__(self, name: str, device: Optional[str] = None) -> None:
        """
        Args:
            name: name of the mask
            device: device of the mask
        Distribution is an abstract class.
        """
        super().__init__(device)
        self.name = name

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Distribution is an abstract class")


class DeterministicMask(Distribution):
    def __init__(
        self,
        mask: torch.Tensor,
        name: str = "deterministic_mask",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            name: name of the mask
            mask: value of the mask of shape `(N,C,H,W)`
            device: device of the mask
        puts a deterministic mask of value mask_value in the stream.
        """
        super().__init__(name, device)
        assert mask.ndim == 4, "mask is expected to be of shape (N,C,H,W)"
        self.mask = mask.to(device)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = x
        output.update({self.name: self.mask})
        return output


class UniformMask(Distribution):
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        name: str = "uniform_mask",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            shape: shape of the mask which is of shape `(N,C,H,W)`
            name: name of the mask
            device: device of the mask inferred if `None`
        samples a uniform mask of shape mask_shape.
        if mask_shape is None, then mask_shape is set to the shape of the input
        at the first call.
        """
        super().__init__(name, device)
        self.shape = shape

    def __call__(self, x) -> Dict[str, torch.Tensor]:
        output = x
        output.update({self.name: torch.rand(size=self.shape, device=self.device)})
        return output


class BernoulliMask(Distribution):
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        name: str = "bernoulli_mask",
        p: Union[float, torch.Tensor] = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            shape: shape of the mask which is of shape `(N,C,H,W)`
            name: name of the mask default to `bernoulli_mask`
            p: probability of the bernoulli distribution
            device: device of the mask inferred if `None`
        samples a bernoulli mask of shape mask_shape.
        if mask_shape is None, then mask_shape is set to the shape of the input
        at the first call.
        """
        super().__init__(name, device)
        self.p = p if isinstance(p, torch.Tensor) else p * torch.ones(size=shape)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = x
        output.update({self.name: torch.bernoulli(self.p).to(self.device)})
        return output


class OneHotCategoricalMask(Distribution):
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        name: str = "negative_onehot_categorical_mask",
        logits: Union[float, torch.Tensor] = 1.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            name: name of the mask
            shape: shape of the mask which is of shape `(N,C,H,W)`
            logits: log probability of the bernoulli distribution
            device: device of the mask
        samples a negative one hot categorical mask of shape mask_shape.
        it is equivalent to sampling a one hot categorical mask and then
        inverting it.
        """
        super().__init__(name, device)
        assert (
            isinstance(logits, float) or logits.shape == shape
        ), "logits must be a float or of shape shape"
        self.shape = shape
        logits = (
            logits
            if isinstance(logits, torch.Tensor)
            else logits * torch.ones(size=self.shape)
        )
        logits = logits.view(1, -1)
        self.distribution = torch.distributions.OneHotCategorical(logits=logits)

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = x
        output.update(
            {self.name: self.distribution.sample().view(self.shape).to(self.device)}
        )
        return output


class NegativeMask(Distribution):
    def __init__(
        self,
        target_name: str,
        name: str = "deterministic_mask",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            name: name of the mask
            target_name: name of the mask to be negated
            device: device of the mask
        negates a mask named target_mask in the stream. i.e. `mask = 1 - target_mask`
        """
        super().__init__(name, device)
        self.target_name = target_name

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.target_name in x, f"{self.target_name} is not in the stream"
        output = x
        output.update({self.name: 1 - x[self.target_name]})
        return output


class NormalMask(Distribution):
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        # noise level is in the official implementation of the paper but
        # it can be shown that it is equivalent to having a different alpha mask
        # for more details see our paper
        # noise_level: float = 0.15,
        # input_scaling: bool = False,
        name: str = "normal_mask",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            shape: shape of the mask of shape `(N,C,H,W)`
            # noise_level: noise level of the mask (removed see paper for the reason)
            # input_scaling: if True, the noise level is scaled by the input `max - min` (removed see our paper for the reason)
            name: name of the mask
            device: device of the mask
        samples a standard normal mask of shape mask_shape.
        if mask_shape is None, then mask_shape is set to the shape of the input
        at the first call.
        the variance of the normal distribution is set to `noise_level * (max(z) - min(z))`
        in the original paper, where `z` input sample, but it can be shown that it is equivalent
        to having a different alpha mask. for more details see our paper.
        """
        super().__init__(name, device)
        self.shape = shape

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = torch.normal(0.0, 1.0, size=self.shape, device=self.device)
        output = x
        output.update({self.name: noise})
        return output


class ResizeMask(Distribution):
    def __init__(
        self,
        source_name: str,
        name="resized_mask",
        target_shape: Optional[Tuple[int, int]] = None,
        target_name: Optional[str] = None,
        mode: str = "bilinear",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            source_name: name of the mask to be resized
            name: name of the resized mask
            target_shape: target shape of the mask to match the shape
            target_name: name of the target mask to match the shape
            mode: mode of the interpolation defaults to `bilinear`
            device: device of the mask inferred if `None`
        resizes the mask to the specified `target_shape` or the shape of the `target_name`.
        if `target_shape` is None, then `target_shape` is set to that of the `target_name`
        at the first call.
        `taget_shape` and `target_name` cannot be both specified.
        """
        super().__init__(name, device)
        assert target_shape is None or target_name is None, (
            "target_shape and target_name cannot be both specified.",
            target_shape,
            target_name,
        )
        assert target_shape is not None or target_name is not None, (
            "target_shape and target_name cannot be both None.",
            target_shape,
            target_name,
        )
        self.target_shape = target_shape
        self.target_name = target_name
        self.mode = mode
        self.source_name = source_name

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.target_shape is None:
            self.target_shape = x[self.target_name].shape[-2:]  # type: ignore checked in __init__ via assert
        if x[self.source_name].shape[-2:] != self.target_shape:
            mask = torch.nn.functional.interpolate(
                x[self.source_name],
                size=self.target_shape,
                mode=self.mode,
                antialias=False,
                align_corners=False,
            )
        else:
            mask = x[self.source_name]
        output = x
        output.update({self.name: mask})
        return output


class RandomCropMask(Distribution):
    def __init__(
        self,
        source_name: str,
        name="cropped_mask",
        target_shape: Optional[Tuple[int, int]] = None,
        target_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            source_name: name of the mask to be cropped
            name: name of the cropped mask
            target_shape: target shape of the mask to match the shape
            target_name: name of the target mask to match the shape
            device: device of the mask inferred if `None`
        crops the mask to the specified `target_shape` or the shape of the `target_name`.
        if `target_shape` is None, then `target_shape` is set to that of the `target_name`
        at the first call.
        `taget_shape` and `target_name` cannot be both specified.
        """
        super().__init__(name, device)
        assert target_shape is None or target_name is None, (
            "target_shape and target_name cannot be both specified.",
            target_shape,
            target_name,
        )
        assert target_shape is not None or target_name is not None, (
            "target_shape and target_name cannot be both None.",
            target_shape,
            target_name,
        )
        self.target_shape = target_shape
        self.target_name = target_name
        self.source_name = source_name
        if self.target_shape is not None:
            self.crop = transforms.RandomCrop(self.target_shape)
        else:
            self.crop = None

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.target_shape is None:
            self.target_shape = x[self.target_name].shape[-2:]  # type: ignore checked in __init__ via assert
            self.crop = transforms.RandomCrop(self.target_shape)

        mask = self.crop(x[self.source_name])  # type: ignore checked in __init__ via assert
        output = x
        output.update({self.name: mask})
        return output


class BlurMask(Distribution):
    def __init__(
        self,
        source_name: str,
        name="blurred_mask",
        kernel_size: int = 5,
        sigma: Union[float, Tuple] = (0.1, 2.0),
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            source_name: name of the mask to be blurred
            name: name of the blurred mask
            kernel_size: kernel size of the gaussian filter
            sigma: sigma of the gaussian filter
            device: device of the mask inferred if `None`
        blurs the mask with a gaussian filter.
        """
        super().__init__(name, device)
        self.kernel_size = kernel_size
        self.blur = transforms.GaussianBlur(kernel_size, sigma)
        self.source_name = source_name

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = self.blur(x[self.source_name])
        output = x
        output.update({self.name: mask})
        return output


class ConvexCombination(Distribution):
    def __init__(
        self,
        name: str = "convex_combination_mask",
        source_mask: str = "input",
        target_mask: str = "baseline",
        alpha_mask: str = "alpha_mask",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            name: name of the interpolated mask defaults to `interp_mask`
            source_mask: name of the source mask defaults to `input`
            target_mask: name of the target mask defaults to `baseline`
            alpha_mask: name of the alpha mask defaults to `alpha_mask`
            device: device of the mask
        interpolates the source mask and the target mask with the alpha mask
        `output = (1-alpha_mask)*source_mask+alpha_mask*target_mask`.
        when alpha is zero, the output is the source mask and when alpha is one,
        the output is the target mask. all masks should have the same spatial shape.
        """
        super().__init__(name, device)
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.alpha_mask = alpha_mask

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert (
            self.source_mask in x and self.target_mask in x and self.alpha_mask in x
        ), (
            "source_mask, target_mask and alpha_mask should be in x",
            self.source_mask in x,
            self.target_mask in x,
            self.alpha_mask in x,
        )
        assert (
            (x[self.source_mask].shape[-2:] == x[self.target_mask].shape[-2:])
            or (x[self.target_mask].shape[-2:] == (1, 1))
        ) and (
            (x[self.alpha_mask].shape[-2:] == x[self.target_mask].shape[-2:])
            or (x[self.alpha_mask].shape[-2:] == (1, 1))
        ), (
            "source_mask, target_mask and alpha_mask should have the same spatial shape",
            x[self.source_mask].shape,
            x[self.target_mask].shape,
            x[self.alpha_mask].shape,
        )
        output = x
        output.update(
            {
                self.name: (1 - x[self.alpha_mask]) * x[self.source_mask]
                +  x[self.alpha_mask]* x[self.target_mask]
            },
        )
        return output

class LinearCombination(Distribution):
    def __init__(self,
                 name: str = "linear_combination_mask",
                 source_mask: str = "input",
                 target_mask: str = "baseline",
                 alpha_source: str = "source_alpha_mask",
                 alpha_target: str = "target_alpha_mask",
                 device: Optional[str] = None,
                 ) -> None:
        """
        Args:
            name: name of the interpolated mask defaults to `interp_mask`
            source_mask: name of the source mask defaults to `input`
            target_mask: name of the target mask defaults to `baseline`
            alpha_source: name of the source alpha mask defaults to `source_alpha_mask`
            alpha_target: name of the target alpha mask defaults to `target_alpha_mask`
            device: device of the mask
        interpolates the source mask and the target mask with the alpha mask
        `output = alpha_source*source_mask+alpha_target*target_mask`.
        """
        super().__init__(name, device)
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.alpha_source = alpha_source
        self.alpha_target = alpha_target

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert (
            self.source_mask in x and self.target_mask in x and self.alpha_source in x and self.alpha_target in x
        ), (
            "source_mask, target_mask, alpha_source and alpha_target should be in x",
            self.source_mask in x,
            self.target_mask in x,
            self.alpha_source in x,
            self.alpha_target in x,
        )
        assert (
            (x[self.source_mask].shape[-2:] == x[self.target_mask].shape[-2:])
            or (x[self.target_mask].shape[-2:] == (1, 1))
        ) and (
            (x[self.alpha_source].shape[-2:] == x[self.source_mask].shape[-2:])
            or (x[self.alpha_source].shape[-2:] == (1, 1))
        ) and (
            (x[self.alpha_target].shape[-2:] == x[self.target_mask].shape[-2:])
            or (x[self.alpha_target].shape[-2:] == (1, 1))
        ), (
            "source_mask, target_mask, alpha_source and alpha_target should have the same spatial shape",
            x[self.source_mask].shape,
            x[self.target_mask].shape,
            x[self.alpha_source].shape,
            x[self.alpha_target].shape,
        )
        output = x
        output.update(
            {
                self.name: x[self.alpha_source] * x[self.source_mask]
                +  x[self.alpha_target]* x[self.target_mask]
            },
        )
        return output

class Compose(Distribution):
    def __init__(
        self,
        distributions: list[Distribution],
        name: str = "compose",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            distributions: list of distributions
            name: name of the mask
            device: device of the mask
        composes a list of distributions.
        """
        super().__init__(name, device)
        self.distributions = distributions

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for t in self.distributions:
            x = t(x)
        return x

    def __getitem__(self, index: int) -> Distribution:
        return self.distributions[index]
