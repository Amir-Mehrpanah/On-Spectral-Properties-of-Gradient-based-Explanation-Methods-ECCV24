import os
import PIL
from matplotlib import pyplot as plt
import numpy as np
import argparse
from pandas import Series
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)


def preprocess(x, img_size):
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=img_size,
        width=img_size,
    )(x)
    x = jnp.array(x)
    x = jnp.expand_dims(x, axis=0) / 255.0
    return x


def save_axis(names, fig, axes, save_output_dir):
    plt.draw()
    for ax, name in zip(axes.flatten(), names):
        extent = (
            ax.get_tightbbox(fig.canvas.renderer)
            .transformed(fig.dpi_scale_trans.inverted())
            .padded(0.05)
        )

        path = os.path.join(save_output_dir, f"{name}.pdf")
        fig.savefig(path, bbox_inches=extent, transparent=True)


# move to visualization.py
def plot_masks(masks, titles, imshow_args={}, ncols=5):
    ncols = ncols
    nrows = np.ceil(len(masks) / ncols).astype(int)
    scale_factor = 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * scale_factor, nrows * scale_factor)
    )
    for i, ax in enumerate(axes.flatten()):
        if i >= len(masks):
            break
        mask = masks.iloc[i]
        ax.imshow(mask, **imshow_args)
        ax.set_title(titles.iloc[i])

    return fig, axes


def preprocess_masks(masks, preprocesses=[]):
    for process in preprocesses:
        masks = masks.apply(process)
    return masks


# def fisher_information(dataframe, prior):
#     '''
#         the wrong way:
#         $Var_y[E_x[\nabla\log y f(x)]]$ where $yf(x) = p(y|x)$
#     '''
#     temp = dataframe.loc[:, "data_path"].apply(np.load)
#     e = temp.loc["meanx"]
#     e = np.stack(e, axis=0)
#     e2 = e**2

#     assert e.shape[0] == prior.shape[0]
#     e2q = (e2 * prior).sum(axis=0)
#     eq2 = (e * prior).sum(axis=0) ** 2
#     fisher = e2q - eq2
#     return fisher


def fisher_information(dynamic_meanx2, static_meanx2, prior):
    """
    $FI = E_x[Var_y[\nabla\log y f(x)]]$ where $yf(x) = p(y|x)$
    $FI = A - B$ where
    $A = \sum_i^k E_x[(\nabla (\log f(x))_i)^2] q_i$
    $B = E_x[(\sum_i^k \nabla (\log f(x))_i q_i)^2]
    """
    static_meanx2 = static_meanx2.loc[:, "data_path"].apply(np.load)
    static_meanx2 = np.stack(static_meanx2, axis=0)

    dynamic_meanx2 = dynamic_meanx2.loc[:, "data_path"].apply(np.load)
    dynamic_meanx2 = dynamic_meanx2.to_numpy()[0]
    assert static_meanx2.shape[0] == prior.shape[0]
    e2q = (static_meanx2 * prior).sum(axis=0)
    eq2 = dynamic_meanx2
    return e2q - eq2


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


def single_query_imagenet(dataset_dir, image_index, input_shape):
    args = argparse.Namespace(
        dataset_dir=dataset_dir, input_shape=input_shape, image_index=[image_index]
    )
    query_imagenet(args)
    return args.image[0], args.label[0], args.image_path[0]


def query_imagenet(args):
    args.image = []
    args.label = []
    args.image_path = []
    image_height = args.input_shape[1]  # (N, H, W, C)
    dataset = tfds.folder_dataset.ImageFolder(root_dir=args.dataset_dir)
    dataset = dataset.as_dataset(split="val", shuffle_files=False)
    skip = args.image_index[0]
    take = args.image_index[1]
    args.image_index = []
    dataset = dataset.skip(skip)
    dataset = dataset.take(take)
    iterator = dataset.as_numpy_iterator()
    logger.info(f"dataset size is {dataset.cardinality()}")
    for i, base_stream in enumerate(iterator, skip):
        base_stream["image"] = preprocess(base_stream["image"], image_height)
        base_stream["label"] = int(base_stream["label"])
        args.image_index.append(i)
        args.image.append(base_stream["image"])
        args.label.append(base_stream["label"])
        args.image_path.append(base_stream["image/filename"].decode())

        assert args.image[-1].shape == args.input_shape


def load_images(image_paths: Series, img_size):
    image_paths = image_paths.apply(PIL.Image.open)
    image_paths = image_paths.apply(
        lambda x: preprocess(x, img_size=img_size).squeeze()
    )
    return image_paths
