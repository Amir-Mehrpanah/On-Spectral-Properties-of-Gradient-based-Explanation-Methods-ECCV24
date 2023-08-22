import jax
import argparse
from tqdm import tqdm

from source import configs
from source.explanation_methods import gather_stats, noise_interpolation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="noise_interpolation",
    choices=["noise_interpolation"],
)
args = parser.parse_args()

if args.method == "noise_interpolation":
    kwargs = {
        "alpha": configs.NoiseInterpolation.alpha,
        "forward": configs.resnet50_forward,
        "num_classes": configs.num_classes,
        "input_shape": configs.input_shape,
        "image": configs.base_stream["image"],
        "label": configs.base_stream["label"],
    }
    abstract_process = noise_interpolation(**kwargs)

    kwargs["method"] = "noise_interpolation"
    kwargs["num_samples"] = configs.NoiseInterpolation.num_samples
    kwargs["batch_size"] = configs.batch_size
    kwargs["seed"] = configs.seed

    del kwargs["forward"]
    del kwargs["image"]
    del kwargs["label"]
    print(kwargs)

stats = gather_stats(
    kwargs["seed"],
    abstract_process,
    kwargs["batch_size"],
    kwargs["num_samples"],
    kwargs["input_shape"],
    kwargs["num_classes"],
)

print("sampling finished")
