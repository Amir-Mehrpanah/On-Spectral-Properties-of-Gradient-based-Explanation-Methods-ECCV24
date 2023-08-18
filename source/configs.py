from collections import namedtuple
import jax
import flaxmodels as fm
import jax.numpy as jnp
import numpy as np
from functools import partial
import tensorflow_datasets as tfds
import os
import sys

sys.path.append(os.getcwd())
from source.operations import preprocess


# general
Stream = namedtuple("Stream", ["name", "statistic"])


class StreamNames:
    vanilla_grad_mask = 10
    results_at_projection = 11
    log_probs = 12


class Statistics:
    mean = 0
    var = 1


seed = 0
batch_size = 32
num_classes = 1000
input_shape = (1, 224, 224, 3)


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
tensorboard_dir = "/local_storage/users/amirme/tensorboard_logs/"
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
    num_samples = 8192
