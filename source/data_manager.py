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


def query_imagenet(args):
    dataset = tfds.folder_dataset.ImageFolder(root_dir=args.dataset_dir)
    dataset = dataset.as_dataset(split="val", shuffle_files=False)
    dataset = dataset.skip(args.image_index)
    base_stream = next(dataset.take(1).as_numpy_iterator())

    image_height = args.input_shape[1]  # (N, H, W, C)
    base_stream["image"] = preprocess(base_stream["image"], image_height)

    args.image = base_stream["image"]
    args.label = base_stream["label"]
    args.image_path = str(base_stream["image/filename"])

    print(f"image_path: {args.image_path}")
    print(f"label: {args.label}")
    raise
