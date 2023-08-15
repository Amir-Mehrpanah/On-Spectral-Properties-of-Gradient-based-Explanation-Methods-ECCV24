import jax
import flaxmodels as fm
import jax.numpy as jnp
from functools import partial
from omegaconf import OmegaConf

resnet50 = fm.ResNet50(
    output="log_softmax",
    pretrained="imagenet",
)
params = resnet50.init(
    jax.random.PRNGKey(0),
    jnp.ones((1, 224, 224, 3)),
)
resnet50_forward = partial(
    resnet50.apply,
    params,
    train=False,
)
base_key = jax.random.PRNGKey(0)


class NoiseInterpolation:
    alpha = 0.0
