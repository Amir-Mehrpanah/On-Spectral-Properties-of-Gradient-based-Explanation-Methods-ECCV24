import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from typing import Dict
import jax
from functools import partial
import jax.numpy as jnp
import tensorflow as tf
import flaxmodels as fm
from PIL import Image

from source import labels

img_size = 224
img_channels = 3


# functions copied from https://github.com/matthias-wright/flaxmodels/blob/main/training/resnet/training.py
def train_prerocess(x):
    x = tf.image.random_crop(x, size=(img_size, img_size, img_channels))
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_flip_left_right(x)
    # Cast to float because if the image has data type int, the following augmentations will convert it
    # to float then apply the transformations and convert it back to int.
    x = tf.cast(x, dtype="float32")
    x = tf.image.random_brightness(x, max_delta=0.5)
    x = tf.image.random_contrast(x, lower=0.1, upper=1.0)
    x = tf.image.random_hue(x, max_delta=0.5)
    x = (x - 127.5) / 127.5
    return x


def val_prerocess(x):
    x = tf.expand_dims(x, axis=0)
    # x = tf.keras.layers.experimental.preprocessing.CenterCrop(
    #     height=img_size, width=img_size
    # )(x)
    x = tf.image.random_crop(x, size=(x.shape[0], img_size, img_size, img_channels))
    x = tf.squeeze(x, axis=0)
    x = tf.cast(x, dtype="float32")
    x = (x - 127.5) / 127.5
    return x


def val_process_simple(x):
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=img_size,
        width=img_size,
    )(x)
    x = jnp.array(x)
    x = jnp.expand_dims(x, axis=0) / 255.0
    return x


@partial(jax.jit, static_argnames=("forward"))
def forward_and_project(forward, x, projection):
    assert x.ndim == 4, "x should be a batch of images"
    log_prob = forward(x)
    return (log_prob @ projection).squeeze(), log_prob


class TestAssests:
    # original paths
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG"
    transforms = val_process_simple
    images: Dict[int, jax.Array] = {
        95: transforms(Image.open("tests/assets/ILSVRC2012_val_00048840.JPEG")),
        96: transforms(Image.open("tests/assets/ILSVRC2012_val_00048864.JPEG")),
        97: transforms(Image.open("tests/assets/ILSVRC2012_val_00049585.JPEG")),
    }
    labels = {
        95: labels.IMAGENET_LABELS_TO_IDX["goldfish, Carassius auratus"],
    }

    batch = jnp.concatenate(list(images.values()), axis=0)
    key = jax.random.PRNGKey(0)
    model = fm.ResNet50(
        output="log_softmax",
        pretrained="imagenet",
    )
    projection = np.zeros((1000, 1))
    projection[1] = 1
    params = model.init(key, images[95])
    forward = partial(
        model.apply,
        params,
        train=False,
    )


class TestResnet50(TestAssests):
    def test_forward_and_project(self):
        projected_log_prob, log_prob = forward_and_project(
            self.forward,
            self.images[95],
            self.projection,
        )
        assert projected_log_prob.shape == ()
        assert log_prob.shape == (1, 1000)
        assert np.exp(projected_log_prob) > 0.9

    def test_grad_shape(self):
        grad_fn = jax.grad(
            forward_and_project,
            argnums=1,
            has_aux=True,
        )

        grad, log_probs = grad_fn(
            self.forward,
            self.images[95],
            self.projection,
        )
        assert grad.shape == self.images[95].shape

    def test_grad_shape_batch(self):
        vgrad_fn = jax.vmap(
            jax.grad(
                forward_and_project,
                argnums=1,
                has_aux=True,
            ),
            in_axes=(None, 0, None),
        )
        vgrad, vlog_prob = vgrad_fn(
            self.forward,
            jnp.expand_dims(self.batch, axis=1),
            self.projection,
        )

        vgrad = jnp.squeeze(vgrad, axis=1)
        vlog_prob = jnp.squeeze(vlog_prob, axis=1)

        assert vgrad.shape == self.batch.shape
        assert vlog_prob.shape == (self.batch.shape[0],1000)
