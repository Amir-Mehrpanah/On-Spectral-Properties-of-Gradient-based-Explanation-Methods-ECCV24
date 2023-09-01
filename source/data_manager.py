import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp


def preprocess(x, img_size):
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=img_size,
        width=img_size,
    )(x)
    x = jnp.array(x)
    x = jnp.expand_dims(x, axis=0) / 255.0
    return x


def minmax_normalize(x):
    x = x - jnp.min(x)
    x = x / jnp.max(x)
    return x


def symmetric_minmax_normalize(x):
    x = x / jnp.max(jnp.abs(x))
    return x


def sum_channels(x):
    x = jnp.sum(x, axis=-1, keepdims=True)  # (H, W, C) -> (N, H, 1)
    return x


def query_imagenet(args):
    dataset = tfds.folder_dataset.ImageFolder(root_dir=args.dataset_dir)
    dataset = dataset.as_dataset(split="val", shuffle_files=False)
    dataset = dataset.skip(args.image_index)
    base_stream = dataset.take(1).as_numpy_iterator().next()

    image_height = args.input_shape[1]  # (N, H, W, C)
    base_stream["image"] = preprocess(base_stream["image"], image_height)

    args.image = base_stream["image"]
    args.label = base_stream["label"]
    args.image_path = base_stream["image/filename"].decode()

    assert args.image.shape == args.input_shape
