import flaxmodels as fm
import jax
import jax.numpy as jnp
from functools import partial


def forward_with_projection(inputs, projection, forward):
    assert inputs.ndim == 4, "inputs should be a batch of images"
    assert inputs.shape[0] == 1, "batch size must match"
    log_prob = forward(inputs)
    results_at_projection = (log_prob @ projection).squeeze()
    return results_at_projection, (results_at_projection, log_prob)


def forward_with_argmax(inputs, forward):
    assert inputs.ndim == 4, "inputs should be a batch of images"
    assert inputs.shape[0] == 1, "batch size must match"
    log_prob = forward(inputs)
    max_index = log_prob.argmax()
    return log_prob[max_index], (log_prob[max_index], log_prob)


def init_resnet50_forward(args):
    resnet50 = fm.ResNet50(
        output="log_softmax",
        pretrained="imagenet",
    )
    params = resnet50.init(
        jax.random.PRNGKey(0),
        jnp.empty(args.input_shape, dtype=jnp.float32),
    )
    resnet50_forward = partial(
        resnet50.apply,
        params,
        train=False,
    )

    args.forward = resnet50_forward
