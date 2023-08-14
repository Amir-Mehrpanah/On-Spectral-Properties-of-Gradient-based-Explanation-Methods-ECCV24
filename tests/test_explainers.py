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
from source import labels
from source import operations


class TestAssests:
    img_size = 224
    # original paths
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG"
    transforms = operations.preprocess
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
        projected_log_prob, log_prob = operations.forward_with_projection(
            name="test",
            projection_name="proj_test",
            forward=self.forward,
            stream={"proj_test": self.projection},
            key=self.key,
        ).concretize()(
            self.images[95],
        )
        assert projected_log_prob.shape == ()
        assert log_prob.shape == (1, 1000)
        assert np.exp(projected_log_prob) > 0.9

    # def test_grad_shape(self):
    #     grad_fn = jax.grad(
    #         static_project,
    #         argnums=1,
    #         has_aux=True,
    #     )

    #     grad, log_probs = grad_fn(
    #         self.forward,
    #         self.images[95],
    #         self.projection,
    #     )
    #     assert grad.shape == self.images[95].shape

    # def test_grad_shape_batch(self):
    #     vgrad_fn = jax.vmap(
    #         jax.grad(
    #             forward_and_project,
    #             argnums=1,
    #             has_aux=True,
    #         ),
    #         in_axes=(None, 0, None),
    #     )
    #     vgrad, vlog_prob = vgrad_fn(
    #         self.forward,
    #         jnp.expand_dims(self.batch, axis=1),
    #         self.projection,
    #     )

    #     vgrad = jnp.squeeze(vgrad, axis=1)
    #     vlog_prob = jnp.squeeze(vlog_prob, axis=1)

    #     assert vgrad.shape == self.batch.shape
    #     assert vlog_prob.shape == (self.batch.shape[0], 1000)

    # def get_concrete_grad_stream_sampler(base_stream, keys):
    #     base_abstract_processes = get_abstract_stream_sampler(base_stream, keys)
    #     base_abstract_processes.append(
    #         operations.vanilla_grad(
    #             name="vanilla_grad_mask",
    #             source_name="convex_combination_mask",
    #             forward=model.apply,
    #         )
    #     )
    #     concrete_processes = operations.concretize(
    #         abstract_processes=base_abstract_processes
    #     )
    #     # create a concrete sequential process
    #     concrete_sequential_process = operations.sequential_call(
    #         concrete_processes=concrete_processes
    #     ).concretize()
    #     return concrete_sequential_process
