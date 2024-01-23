import flaxmodels as fm
import jax
import logging
import jax.numpy as jnp
from functools import partial

logger = logging.getLogger(__name__)

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


def init_resnet50_randomized_forward(args):
    # choices: ['Conv_0', 'BatchNorm_0', 'Bottleneck_0', 'Bottleneck_1', 'Bottleneck_2', 'Bottleneck_3', 'Bottleneck_4', 'Bottleneck_5', 'Bottleneck_6', 'Bottleneck_7', 'Bottleneck_8', 'Bottleneck_9', 'Bottleneck_10', 'Bottleneck_11', 'Bottleneck_12', 'Bottleneck_13', 'Bottleneck_14', 'Bottleneck_15', 'Dense_0']
    assert args.layer_randomization is not None, "layer_randomization must be specified"

    resnet50 = fm.ResNet50(
        output=args.output_layer,
        pretrained="imagenet",
    )
    resnet50_random = fm.ResNet50(
        output=args.output_layer,
        pretrained=None,
    )
    params = resnet50.init(
        jax.random.PRNGKey(0),
        jnp.empty(args.input_shape, dtype=jnp.float32),
    )

    assert args.layer_randomization in params["params"].keys(), f"layer_randomization must be one of {params['params'].keys()}"
    
    params_random = resnet50_random.init(
        jax.random.PRNGKey(0),
        jnp.empty(args.input_shape, dtype=jnp.float32),
    )
    # Randomize the weights of the last layer
    params["params"][args.layer_randomization] = params_random["params"][args.layer_randomization]
    logger.info(f"Randomized layer: {args.layer_randomization}")
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
