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

jax.config.update("jax_log_compiles", True)

sys.path.append(os.getcwd())
from source import labels
from source import operations
from source import explainers
from test_operations import get_abstract_stream_sampler


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
        projected_log_prob, (_, log_prob) = explainers.forward_with_projection(
            projection_name="proj_test",
            forward=self.forward,
            stream={"proj_test": self.projection},
        ).concretize()(
            self.images[95],
        )
        assert projected_log_prob.shape == ()
        assert log_prob.shape == (1, 1000)
        assert np.exp(projected_log_prob) > 0.9

    def test_grad_shape(self):
        concrete_forward_with_projection = explainers.forward_with_projection(
            projection_name="proj_test",
            forward=self.forward,
            stream={"proj_test": self.projection},
        ).concretize()
        grad_fn = jax.grad(
            concrete_forward_with_projection,
            has_aux=True,
        )

        grad, (_, log_probs) = grad_fn(
            self.images[95],
        )
        assert grad.shape == self.images[95].shape

    def test_grad_shape_batch(self):
        concrete_forward_with_projection = explainers.forward_with_projection(
            projection_name="test_proj",  # static projection
            forward=self.forward,
        ).concretize()
        vgrad_fn = jax.jit(
            jax.vmap(
                jax.grad(
                    concrete_forward_with_projection,
                    has_aux=True,
                ),
                in_axes=(0, None),
            )
        )
        vgrad, (_, log_prob) = vgrad_fn(
            jnp.expand_dims(self.batch, axis=1),
            {"test_proj": self.projection},
        )
        vgrad = jnp.squeeze(vgrad, axis=1)
        assert vgrad.shape == self.batch.shape

    def get_concrete_grad_stream_sampler(self, base_stream, keys):
        base_stream["proj_test"] = self.projection
        base_abstract_processes = get_abstract_stream_sampler(base_stream, keys)
        concrete_forward_with_projection = explainers.forward_with_projection(
            forward=self.forward,
            projection_name="proj_test",
        ).concretize()
        base_abstract_processes.append(
            explainers.vanilla_gradient(
                name="vanilla_grad_mask",
                source_name="convex_combination_mask",
                concrete_forward_with_projection=concrete_forward_with_projection,
            )
        )
        concrete_processes = operations.concretize_all(
            abstract_processes=base_abstract_processes
        )
        # create a concrete sequential process
        concrete_sequential_process = operations.sequential_call(
            concrete_processes=concrete_processes
        ).concretize()
        return concrete_sequential_process

    def test_grad_stream_sampling(self):
        base_stream = {}
        num_samples = 10
        keys = jax.random.split(self.key, num=num_samples)

        concrete_sequential_process = self.get_concrete_grad_stream_sampler(
            base_stream, keys
        )

        # compute the expected stream
        expected_stream = copy.deepcopy(base_stream)
        concrete_sequential_process(keys[0], expected_stream)

        # vmap the concrete sequential process
        vmap_concrete_sequential_process = jax.vmap(
            concrete_sequential_process, in_axes=(0, None)
        )

        # count the number of compilations
        vmap_concrete_sequential_process = operations.count_compilations(
            vmap_concrete_sequential_process
        )
        # compile the concrete sequential process and call it
        compiled_concrete_sequential_process = jax.jit(vmap_concrete_sequential_process)
        result_stream = compiled_concrete_sequential_process(keys, base_stream)

        assert vmap_concrete_sequential_process.number_of_compilations == 1
        assert base_stream is not result_stream
        assert result_stream.keys() == expected_stream.keys()
        assert (
            result_stream["vanilla_grad_mask"][0]
            != result_stream["vanilla_grad_mask"][1]
        ).all()
        assert result_stream["vanilla_grad_mask"].shape[0] == num_samples
        assert (
            result_stream["vanilla_grad_mask"].shape[1:]
            == expected_stream["vanilla_grad_mask"].shape
        )
        np.testing.assert_allclose(
            result_stream["vanilla_grad_mask"][0],
            expected_stream["vanilla_grad_mask"],
            atol=1e-2,
        )
