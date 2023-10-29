import jax.numpy as jnp
import jax
import numpy as np
import logging
import sys
import os

sys.path.append(os.getcwd())
from source.utils import AbstractFunction, debug_nice

logger = logging.getLogger(__name__)


@AbstractFunction
def _measure_inconsistency_cosine_distance(
    batch_mean: jnp.ndarray,
    downsampling_factor,
    downsampling_method,
):
    assert batch_mean.ndim == 5, (
        "image batched group should be 5D (B,A,H,W,C) "
        "where B is the batch size and A is the number of columns in pivot table."
    )
    B, T, H, W, _ = batch_mean.shape
    new_H = H // downsampling_factor
    new_W = W // downsampling_factor
    downsampled: jax.Array = jax.image.resize(
        batch_mean,
        shape=(
            B,
            T,
            new_H,
            new_W,
            1,  # collapse the color channels
        ),
        method=downsampling_method,
    )

    downsampled = jnp.squeeze(downsampled, axis=-1)
    downsampled_0 = downsampled[:, [0], ...].reshape((B, 1, -1))
    downsampled_gt0 = downsampled[:, 1:, ...].reshape((B, T - 1, -1))

    norm_0 = jnp.linalg.norm(downsampled_0, axis=-1, keepdims=True)
    norm_gt0 = jnp.linalg.norm(downsampled_gt0, axis=-1, keepdims=True)

    average_cosine_similarity = jnp.einsum(
        "bti,btj->b",
        downsampled_0 / norm_0,
        downsampled_gt0 / norm_gt0,
    ) / (new_H * new_W * (T - 1))

    return 1 - average_cosine_similarity


@AbstractFunction
def _measure_inconsistency_DSSIM(
    batch_mean, batch_meanx2, c1, c2, downsampling_factor, downsampling_method
):
    """
    computes the DSSIM between two images
    DSSIM = (1-SSIM)/2
    SSIM stands for structural similarity index measure
    """
    B, T, H, W, _ = batch_mean.shape
    new_H = H // downsampling_factor
    new_W = W // downsampling_factor
    batch_mean: jax.Array = jax.image.resize(
        batch_mean,
        shape=(
            B,
            T,
            new_H,
            new_W,
            1,  # collapse the color channels
        ),
        method=downsampling_method,
    )
    batch_meanx2: jax.Array = jax.image.resize(
        batch_meanx2,
        shape=(
            B,
            T,
            new_H,
            new_W,
            1,  # collapse the color channels
        ),
        method=downsampling_method,
    )
    batch_mean = np.squeeze(batch_mean, axis=-1)
    batch_meanx2 = np.squeeze(batch_meanx2, axis=-1)
    sigma2 = batch_meanx2 - batch_mean * 2
    sigma = np.sqrt(sigma2)

    l = (2 * batch_mean.prod(axis=1) + c1) / ((batch_mean**2).sum(axis=1) + c1)
    c = (2 * sigma.prod(axis=1) + c2) / (sigma2.sum(axis=1) + c2)
    # s = 0 we assume that sigmaxy is zero for all pairs of images
    ssim = l * c
    dssim = (1 - ssim) / 2
    dssim = dssim.mean(axis=(2, 3))
    assert dssim.shape == (B,)
    return dssim


def measure_inconsistency(numpy_iterator, concrete_inconsistency_measure):
    results = {"inconsistency": []}
    for batch in numpy_iterator:
        data = batch.pop("data")
        logger.debug(
            f"computing inconsistency for {debug_nice(data)} with {debug_nice(concrete_inconsistency_measure)}"
        )
        inconsistency = concrete_inconsistency_measure(*data)
        results["inconsistency"].append(inconsistency)
        for k, v in batch.items():
            if k not in results:
                results[k] = []
            results[k].append(v)  # other keys are indices
    for k in results:
        logger.debug(f"concatenating {k}, {debug_nice(results[k])}")
        results[k] = np.concatenate(results[k])
    return results
