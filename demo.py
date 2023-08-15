import jax
from source.explanation_methods import noise_interpolation
import argparse
from source import configs


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="noise_interpolation",
    choices=["noise_interpolation"],
)
args = parser.parse_args()
key = configs.base_key
stream = {"image": configs.images[0]}

if args.method == "noise_interpolation":
    noise_interpolation(
        alpha=configs.NoiseInterpolation.alpha,
        forward=configs.resnet50_forward,
    )
    concrete_process = noise_interpolation.concretize()


concrete_process = jax.jit(concrete_process)
stream = concrete_process(key, stream)
