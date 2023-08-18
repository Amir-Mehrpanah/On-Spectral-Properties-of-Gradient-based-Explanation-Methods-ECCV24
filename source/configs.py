import jax
import flaxmodels as fm
import jax.numpy as jnp
from functools import partial
import tensorflow_datasets as tfds
import os
import sys

sys.path.append(os.getcwd())
from source.operations import preprocess


# general
class Streams:
    vanilla_grad_mask = 0
    results_at_projection = 1
    log_probs = 2


class Statistics:
    mean = 0
    var = 1


base_key = jax.random.PRNGKey(0)
sampling_batch_size = 128
num_classes = 1000
input_shape = (1, 224, 224, 3)
stats = {
    (Streams.vanilla_grad_mask, Statistics.mean): jnp.zeros(shape=input_shape),
    (Streams.vanilla_grad_mask, Statistics.var): jnp.zeros(shape=input_shape),
    (Streams.results_at_projection, Statistics.mean): jnp.zeros(shape=()),
    (Streams.results_at_projection, Statistics.var): jnp.zeros(shape=()),
    (Streams.log_probs, Statistics.mean): jnp.zeros(shape=(1, num_classes)),
    (Streams.log_probs, Statistics.var): jnp.zeros(shape=(1, num_classes)),
}

# model
resnet50 = fm.ResNet50(
    output="log_softmax",
    pretrained="imagenet",
)
params = resnet50.init(
    jax.random.PRNGKey(0),
    jnp.empty(input_shape, dtype=jnp.float32),
)
resnet50_forward = partial(
    resnet50.apply,
    params,
    train=False,
)


# data
datadir = "/local_storage/datasets/imagenet/"
image_height = input_shape[1]
dataset_skip_index = 0
dataset = tfds.folder_dataset.ImageFolder(root_dir=datadir)
dataset = dataset.as_dataset(split="val", shuffle_files=False)
dataset = dataset.skip(dataset_skip_index)
base_stream = next(dataset.take(1).as_numpy_iterator())
base_stream["image"] = preprocess(base_stream["image"], image_height)
del base_stream["image/filename"]


# explanation methods
class NoiseInterpolation:
    alpha = 0.1
    num_samples = 2048
    num_batches = num_samples // sampling_batch_size
