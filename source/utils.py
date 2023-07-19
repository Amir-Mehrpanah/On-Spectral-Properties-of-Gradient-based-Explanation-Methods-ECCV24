from copy import deepcopy
from glob import glob
import pickle
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple, Callable, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


class SmartDevice:
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

    def to(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in x.items()}


class Statistic:
    def __init__(
        self,
        num_samples: int,
        input_shape: Union[Tuple[int, int, int], Tuple[int]],
    ) -> None:
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.n = 0

    def update(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[1:] == self.input_shape
        ), "input tensor is expected to be of shape (N,{}), got {}".format(
            self.input_shape, x.shape
        )
        self.n += x.shape[0]
        return torch.empty(0)

    def get_statistic(self) -> torch.Tensor:
        assert (
            self.n == self.num_samples
        ), "Not enough samples, got {}, expected {}".format(self.n, self.num_samples)
        return torch.empty(0)


class Collect(Statistic):
    def __init__(
        self,
        num_samples: int,
        loc: Tuple,
        input_shape: Union[Tuple[int, int, int], Tuple[int]],
    ) -> None:
        super().__init__(
            num_samples,
            input_shape,
        )
        self.exps = []
        self.loc = loc

    def update(self, x: torch.Tensor):
        super().update(x)
        self.exps.append(x[self.loc])

    def get_statistic(self):
        super().get_statistic()
        return torch.stack(self.exps, dim=1)


class Mean(Statistic):
    def __init__(
        self,
        num_samples: int,
        input_shape: Union[Tuple[int, int, int], Tuple[int]],
    ) -> None:
        super().__init__(num_samples, input_shape)
        self.mean = torch.zeros(input_shape)

    def update(self, x: torch.Tensor):
        super().update(x)
        self.mean += torch.sum(x, dim=0) / self.num_samples

    def get_statistic(self):
        super().get_statistic()
        return self.mean


class Variance(Statistic):
    def __init__(
        self,
        num_samples: int,
        input_shape: Union[Tuple[int, int, int], Tuple[int]],
    ) -> None:
        super().__init__(num_samples, input_shape)
        self.mean = torch.zeros(input_shape)
        self.mean2 = torch.zeros(input_shape)

    def update(self, x: torch.Tensor):
        super().update(x)
        self.mean = torch.sum(x, dim=0) / self.num_samples
        self.mean2 = torch.sum(x * x, dim=0) / self.num_samples

    def get_statistic(self):
        super().get_statistic()
        return self.mean2 - self.mean**2


class Std(Variance):
    def get_statistic(self):
        return torch.sqrt(super().get_statistic())


class Statistics:
    def __init__(
        self,
        stats: Dict[str, Statistic],
    ) -> None:
        self.stats = deepcopy(stats)

    def update_statistics(self, explanation: torch.Tensor):
        for key in self.stats.keys():
            self.stats[key].update(explanation)

    def get_statistics(self):
        return {
            key: self.stats[key].get_statistic().cpu().numpy()
            for key in self.stats.keys()
        }


def print_dict_str_tensor_shapes(dict_str_tensor: Dict[str, torch.Tensor]) -> None:
    print(
        dict(
            map(
                lambda x: (x[0], None if x[1] is None else x[1].shape),
                dict_str_tensor.items(),
            )
        )
    )


def print_dict_str_tensor_devices(dict_str_tensor: Dict[str, torch.Tensor]) -> None:
    print(
        dict(
            map(
                lambda x: (x[0], None if x[1] is None else x[1].device),
                dict_str_tensor.items(),
            )
        )
    )


def print_dict_str_tensor_func(
    dict_str_tensor: Dict[str, torch.Tensor], func: Callable, skip_none: bool = True
) -> None:
    print(
        dict(
            map(
                lambda x: (x[0], None if x[1] is None and skip_none else func(x[1])),
                dict_str_tensor.items(),
            )
        )
    )


def get_device(model: torch.nn.Module):
    device = str(next(model.parameters()).device)
    return device


def rename_dict(sample: Dict[str, torch.Tensor], keys_to_args_mapping: Dict[str, str]):
    for old_key, new_key in keys_to_args_mapping.items():
        sample[new_key] = sample.pop(old_key)
    return sample


def merge_saliency_dict(saliency_glob_path, save_path):
    saliencies = []
    saliency_paths = glob(saliency_glob_path)
    # load pickle
    print("Loading saliencies...")
    for saliency_path in tqdm(saliency_paths):
        with open(saliency_path, "rb") as f:
            saliencies.append(pickle.load(f))

    # append all saliency
    print("Merging saliencies...")
    merged_saliency = {}
    for saliency in tqdm(saliencies):
        for explanation_name, explanation in saliency.items():
            if explanation_name not in merged_saliency:
                merged_saliency[explanation_name] = {}
            for statistic_name, statistic in explanation.items():
                if statistic_name == "image":
                    if statistic_name not in merged_saliency[explanation_name]:
                        merged_saliency[explanation_name][statistic_name] = statistic
                    continue
                if statistic_name not in merged_saliency[explanation_name]:
                    merged_saliency[explanation_name][statistic_name] = []
                merged_saliency[explanation_name][statistic_name].append(statistic)

    print("Calculating mean...")
    for explanation_name, explanation in tqdm(merged_saliency.items()):
        for statistic_name, statistic in explanation.items():
            if statistic_name == "image":
                continue
            merged_saliency[explanation_name][statistic_name] = np.stack(
                statistic, axis=0
            ).mean(axis=0)

    print("Saving...")
    with open(save_path, "wb") as f:
        pickle.dump(merged_saliency, f)


# create baselines
# blur baseline
blur_baseline = transforms.GaussianBlur(kernel_size=5, sigma=1)
# black baseline
black_16x16_tiled_baseline = torch.zeros(size=(1, 1, 16, 16))
black_12x12_tiled_baseline = torch.zeros(size=(1, 1, 16, 16))
black_8x8_tiled_baseline = torch.zeros(size=(1, 1, 8, 8))
black_3x3_tiled_baseline = torch.zeros(size=(1, 1, 3, 3))

named_locations = {
    "A": (slice(None), 125, 90),
    "B": (slice(None), 210, 175),
    "C": (slice(None), 25, 25),
    "D": (slice(None), 100, 115),
    "E": (slice(None), 125, 200),
    "F": (slice(None), 50, 200),
    "G": (slice(None), 90, 125),
}

named_regions: Dict[
    str, Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]
] = {
    "A": (slice(None), slice(120, 130), slice(85, 95)),
    "B": (slice(None), slice(205, 215), slice(170, 180)),
    "C": (slice(None), slice(20, 30), slice(20, 30)),
    "D": (slice(None), slice(75, 85), slice(110, 120)),
    "E": (slice(None), slice(120, 130), slice(195, 205)),
    "F": (slice(None), slice(45, 55), slice(195, 205)),
    "G": (slice(None), slice(150, 160), slice(20, 30)),
    # "W": (slice(None), slice(0, 244), slice(0, 244)),
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Things you can do directly from the command line with this script"
    )
    parser.add_argument(
        "--saliency_glob_path",
        type=str,
        required=True,
        help="Path to the saliency pickle file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the output plot",
    )
    args = parser.parse_args()
    saliency_glob_path = args.saliency_glob_path
    save_path = args.save_path

    merge_saliency_dict(
        saliency_glob_path=saliency_glob_path,
        save_path=save_path,
    )