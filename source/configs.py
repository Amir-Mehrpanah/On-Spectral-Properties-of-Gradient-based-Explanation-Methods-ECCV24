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
base_key = jax.random.PRNGKey(0)
sampling_batch_size = 32

# model
resnet50 = fm.ResNet50(
    output="log_softmax",
    pretrained="imagenet",
)
params = resnet50.init(
    jax.random.PRNGKey(0),
    jnp.ones((1, 224, 224, 3)),
)
resnet50_forward = partial(
    resnet50.apply,
    params,
    train=False,
)


# data
datadir = "/local_storage/datasets/imagenet/"
image_size = 224
dataset_skip_index = 0
dataset = tfds.folder_dataset.ImageFolder(root_dir=datadir)
dataset = dataset.as_dataset(split="val", shuffle_files=False)
dataset = dataset.skip(dataset_skip_index)
stream = next(dataset.take(1).as_numpy_iterator())
stream["image"] = preprocess(stream["image"], image_size)
del stream["image/filename"]


# explanation methods
class NoiseInterpolation:
    alpha = 0.1
    num_samples = 128
    num_batches = num_samples // sampling_batch_size
