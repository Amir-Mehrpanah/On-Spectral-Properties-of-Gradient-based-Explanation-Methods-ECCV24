import copy
import os
import sys
from typing import Dict
import numpy as np
import jax
from functools import partial
import jax.numpy as jnp
import flaxmodels as fm
from PIL import Image

sys.path.append(os.getcwd())
from source import operations, helpers, explainers, labels
from test_operations import get_abstract_stream_sampler


class TestAssests:
    img_size = 224
    # original paths
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG"
    transforms = helpers.preprocess
    images: Dict[int, jax.Array] = {
        95: transforms(
            Image.open("tests/assets/ILSVRC2012_val_00048840.JPEG"), img_size=img_size
        ),
        96: transforms(
            Image.open("tests/assets/ILSVRC2012_val_00048864.JPEG"), img_size=img_size
        ),
        97: transforms(
            Image.open("tests/assets/ILSVRC2012_val_00049585.JPEG"), img_size=img_size
        ),
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


class TestWithResnet50(TestAssests):
    def test_forward_and_project(self):
        projected_log_prob, (_, log_prob) = explainers.forward_with_projection(
            self.images[95],
            projection=self.projection,
            forward=self.forward,
        )
        assert projected_log_prob.shape == ()
        assert log_prob.shape == (1, 1000)
        assert np.exp(projected_log_prob) > 0.9

    def test_grad_shape(self):
        grad_fn = jax.grad(
            explainers.forward_with_projection,
            has_aux=True,
        )

        grad, (_, log_probs) = grad_fn(
            self.images[95],
            projection=self.projection,
            forward=self.forward,
        )
        assert grad.shape == self.images[95].shape

    def test_grad_shape_batch(self):
        vgrad_fn = jax.jit(
            jax.vmap(
                jax.grad(
                    explainers.forward_with_projection,
                    has_aux=True,
                ),
                in_axes=(0, None, None),
            ),
            static_argnums=(2,),
        )
        vgrad, (_, log_prob) = vgrad_fn(
            jnp.expand_dims(self.batch, axis=1),
            self.projection,
            self.forward,
        )

        vgrad = jnp.squeeze(vgrad, axis=1)
        assert vgrad.shape == self.batch.shape

    def disentangled_forward(self, input_noise, projection, forward):
        input, noise = input_noise
        noisy_image = input + noise
        return explainers.forward_with_projection(noisy_image, projection, forward)

