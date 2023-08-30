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
from source import data_manager, explainers


class TestAssests:
    img_size = 224
    # original paths
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG"
    transforms = data_manager.preprocess
    images: Dict[int, jax.Array] = {
        0: transforms(
            Image.open("tests/assets/val/n01491361/ILSVRC2012_val_00026626.JPEG"),
            img_size=img_size,
        ),
        95: transforms(
            Image.open("tests/assets/val/n01491361/ILSVRC2012_val_00048840.JPEG"),
            img_size=img_size,
        ),
        96: transforms(
            Image.open("tests/assets/val/n01491361/ILSVRC2012_val_00048864.JPEG"),
            img_size=img_size,
        ),
        97: transforms(
            Image.open("tests/assets/val/n01491361/ILSVRC2012_val_00049585.JPEG"),
            img_size=img_size,
        ),
    }
    labels = {
        0: 46,
        95: 1,
    }
    projection_ = np.zeros((1000, 1))
    projection_46 = copy.deepcopy(projection_)
    projection_46[46] = 1
    projection_1 = copy.deepcopy(projection_)
    projection_1[1] = 1
    projections = {
        0: projection_46,
        95: projection_1,
    }

    batch = jnp.concatenate(list(images.values()), axis=0)
    key = jax.random.PRNGKey(0)
    model = fm.ResNet50(
        output="log_softmax",
        pretrained="imagenet",
    )
    params = model.init(key, images[95])
    forward = partial(
        model.apply,
        params,
        train=False,
    )


class TestWithResnet50(TestAssests):
    def test_forward_and_project(self):
        projected_log_prob, (_, log_prob) = explainers.forward_with_projection(
            self.images[0],
            projection=self.projections[0],
            forward=self.forward,
        )
        assert projected_log_prob.shape == ()
        assert log_prob.shape == (1, 1000)

    def test_grad_shape(self):
        grad_fn = jax.grad(
            explainers.forward_with_projection,
            has_aux=True,
        )

        grad, (_, log_probs) = grad_fn(
            self.images[95],
            projection=self.projections[95],
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
