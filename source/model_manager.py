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


def init_resnet50_forward(args):
    resnet50 = fm.ResNet50(
        output=args.output_layer,
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
    if hasattr(args, "forward"):
        assert isinstance(
            args.forward, list
        ), f"forward must be a list recieved {type(args.forward)}"
        args.forward.append(resnet50_forward)
    else:
        args.forward = [resnet50_forward]
