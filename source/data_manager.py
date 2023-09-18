from matplotlib import pyplot as plt
import numpy as np
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


# move to visualization.py
def plot_masks(masks, titles, imshow_args={}, ncols=5):
    ncols = ncols
    nrows = len(masks) // ncols
    scale_factor = 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * scale_factor, nrows * scale_factor)
    )
    for i, ax in enumerate(axes.flatten()):
        mask = masks.iloc[i]
        ax.imshow(mask, **imshow_args)
        ax.set_title(titles.iloc[i])


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
    temp = static_meanx2.loc[:, "data_path"].apply(np.load)
    e = temp.loc["meanx2"]
    e2 = temp.loc["meanx2"]
    e = np.stack(e, axis=0)
    e2 = np.stack(e2, axis=0)

    assert e.shape[0] == prior.shape[0]
    e2q = (e2 * prior).sum(axis=0)
    eq2 = (e * prior).sum(axis=0) ** 2
    fisher = e2q - eq2
    return fisher


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
