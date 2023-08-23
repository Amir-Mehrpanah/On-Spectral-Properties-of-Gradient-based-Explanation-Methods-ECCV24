import jax.numpy as jnp
import argparse
import time
from source import configs
from source.explanation_methods import gather_stats, noise_interpolation
from source.helpers import Stream, StreamNames, Statistics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    type=str,
    default="noise_interpolation",
    choices=["noise_interpolation"],
)
args = parser.parse_args()

if args.method == "noise_interpolation":
    input_shape = configs.input_shape
    num_classes = configs.num_classes

    kwargs = {
        "alpha": configs.NoiseInterpolation.alpha,
        "forward": configs.resnet50_forward,
        "num_classes": num_classes,
        "input_shape": input_shape,
        "image": configs.base_stream["image"],
        "label": configs.base_stream["label"],
    }
    abstract_process = noise_interpolation(**kwargs)

    kwargs["method"] = "noise_interpolation"
    kwargs["max_batches"] = configs.NoiseInterpolation.max_batches
    kwargs["batch_size"] = configs.batch_size
    kwargs["seed"] = configs.seed

    del kwargs["forward"]
    del kwargs["image"]
    del kwargs["label"]
    print(kwargs)

batch_index_key = Stream(
    StreamNames.batch_index,
    Statistics.none,
)
monitored_statistic_key = Stream(
    StreamNames.vanilla_grad_mask,
    Statistics.abs_delta,
)
monitored_statistic_source_key = Stream(
    StreamNames.vanilla_grad_mask,
    Statistics.meanx2,
)
stats = {
    Stream(
        StreamNames.vanilla_grad_mask,
        Statistics.meanx,
    ): jnp.zeros(shape=input_shape),
    monitored_statistic_source_key: jnp.zeros(shape=input_shape),
    Stream(
        StreamNames.results_at_projection,
        Statistics.meanx,
    ): jnp.zeros(shape=()),
    Stream(
        StreamNames.results_at_projection,
        Statistics.meanx2,
    ): jnp.zeros(shape=()),
    Stream(
        StreamNames.log_probs,
        Statistics.meanx,
    ): jnp.zeros(shape=(1, num_classes)),
    Stream(
        StreamNames.log_probs,
        Statistics.meanx2,
    ): jnp.zeros(shape=(1, num_classes)),
    monitored_statistic_key: jnp.inf,
    batch_index_key: 0,
}

start = time.time()
stats = gather_stats(
    kwargs["seed"],
    abstract_process,
    kwargs["batch_size"],
    kwargs["max_batches"],
    1e-8,
    stats,
    monitored_statistic_source_key,
    monitored_statistic_key,
    batch_index_key,
)
end = time.time()
print(f"Time: {end - start}s")
print("number of samples", stats[batch_index_key])
