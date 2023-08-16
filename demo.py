import jax
import argparse

from source.explanation_methods import noise_interpolation
from source import configs


parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="noise_interpolation",
    choices=["noise_interpolation"],
)
args = parser.parse_args()
batch_keys = jax.random.split(
    configs.base_key,
    num=configs.NoiseInterpolation.num_batches,
)

if args.method == "noise_interpolation":
    noise_interpolation(
        alpha=configs.NoiseInterpolation.alpha,
        forward=configs.resnet50_forward,
        num_classes=configs.num_classes,
        input_shape=configs.input_shape,
        image=configs.base_stream["image"],
        label=configs.base_stream["label"],
    )
    concrete_process = noise_interpolation.concretize()

compiled_concrete_process = jax.jit(
    jax.vmap(
        concrete_process,
        in_axes=(0),
    )
)

with jax.log_compiles():
    for batch_key in batch_keys:
        sample_keys = jax.random.split(
            batch_key,
            num=configs.sampling_batch_size,
        )
        stream = compiled_concrete_process(sample_keys)
print(stream)