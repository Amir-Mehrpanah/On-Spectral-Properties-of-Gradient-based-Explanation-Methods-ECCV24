import os
from typing import Any
import flaxmodels as fm
from flax.training import train_state
import jax
import logging
import jax.numpy as jnp
from flax.training import checkpoints
from functools import partial

import optax

logger = logging.getLogger(__name__)


def forward_with_projection(inputs, projection, forward):
    assert inputs.ndim == 4, "inputs should be a batch of images"
    assert inputs.shape[0] == 1, "batch size must match"
    log_prob = forward(inputs)
    results_at_projection = (log_prob @ projection).squeeze()
    return results_at_projection, log_prob


def init_resnet50_forward(args):
    ckpt_path = os.path.join(args.save_temp_base_dir, args.dataset)
    logger.debug(f"loading model from: {ckpt_path}")
    if args.dataset == "imagenet":
        logger.debug(f"loading imagenet model with input shape: {args.input_shape}")
        resnet50_forward = init_resnet50_imagenet(args, ckpt_path)
    elif args.dataset == "food101":
        logger.debug(f"loading food101 model with input shape: {args.input_shape}")
        resnet50_forward = init_resnet50_food101(args, ckpt_path)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if hasattr(args, "forward"):
        assert isinstance(
            args.forward, list
        ), f"forward must be a list recieved {type(args.forward)}"
        args.forward.append(resnet50_forward)
    else:
        args.forward = [resnet50_forward]


class TrainState(train_state.TrainState):
    batch_stats: Any
    epoch: int


def init_resnet50_food101(args, ckpt_path):
    model = fm.ResNet50(
        output=args.output_layer,
        pretrained=None,
        num_classes=args.num_classes,
    )
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.ones(args.input_shape),
    )
    tx = optax.sgd(0)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        epoch=0,
    )

    state = checkpoints.restore_checkpoint(ckpt_path, state)

    variables = {
        "params": state.params,
        "batch_stats": state.batch_stats,
    }
    resnet50_forward = partial(
        state.apply_fn,
        variables,
        train=False,
        mutable=False,
    )
    return resnet50_forward


def init_resnet50_imagenet(args, ckpt_path):
    resnet50 = fm.ResNet50(
        output=args.output_layer,
        pretrained="imagenet",
        ckpt_dir=ckpt_path,
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

    return resnet50_forward


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

    assert (
        args.layer_randomization in params["params"].keys()
    ), f"layer_randomization must be one of {params['params'].keys()}"

    params_random = resnet50_random.init(
        jax.random.PRNGKey(0),
        jnp.empty(args.input_shape, dtype=jnp.float32),
    )
    # Randomize the weights of the last layer
    params["params"][args.layer_randomization] = params_random["params"][
        args.layer_randomization
    ]
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
