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

import sys

if "outputs/temp/imagenet/vit/source/vision_transformer" not in sys.path:
    sys.path.append("outputs/temp/imagenet/vit/source/vision_transformer")

from outputs.temp.imagenet.vit.source.vision_transformer.vit_jax import checkpoint
from outputs.temp.imagenet.vit.source.vision_transformer.vit_jax import models
from outputs.temp.imagenet.vit.source.vision_transformer.vit_jax.configs import (
    models as models_config,
)

logger = logging.getLogger(__name__)


def forward_with_projection(inputs, params, projection, forward):
    assert inputs.ndim == 4, "inputs should be a batch of images"
    assert inputs.shape[0] == 1, "batch size must match"
    logger.debug(f"forward with projection")
    log_prob = forward(params, inputs)
    results_at_projection = (log_prob @ projection).squeeze()
    return results_at_projection, log_prob


def init_resnet50_forward(args):
    ckpt_path = os.path.join(args.save_temp_base_dir, args.dataset)
    logger.debug(f"loading model from: {ckpt_path}")
    if args.dataset == "imagenet":
        logger.debug(f"loading imagenet model with input shape: {args.input_shape}")
        resnet50_forward, params = init_resnet50_imagenet(args, ckpt_path)
    elif args.dataset == "food101" or args.dataset == "curated_breast_imaging_ddsm":
        logger.debug(
            f"loading {args.dataset} model with input shape: {args.input_shape}"
        )
        resnet50_forward, params = init_resnet50_non_imagenet(args, ckpt_path)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if hasattr(args, "forward"):
        assert isinstance(
            args.forward, list
        ), f"forward must be a list recieved {type(args.forward)}"
        args.forward.append(resnet50_forward)
        args.params.append(params)
    else:
        args.forward = [resnet50_forward]
        args.params = [params]


class TrainState(train_state.TrainState):
    batch_stats: Any
    epoch: int


def init_resnet50_non_imagenet(args, ckpt_path):
    model = fm.ResNet50(
        output=args.output_layer,
        pretrained=None,
        num_classes=args.num_classes,
        normalize=False,
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
        train=False,
        mutable=False,
    )
    return resnet50_forward, variables


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
        train=False,
    )

    return resnet50_forward, params


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
        train=False,
    )
    if hasattr(args, "forward"):
        assert isinstance(
            args.forward, list
        ), f"forward must be a list recieved {type(args.forward)}"
        args.forward.append(resnet50_forward)
        args.params.append(params)
    else:
        args.forward = [resnet50_forward]
        args.params = [params]


def init_vit_forward(args):
    ckpt_path = os.path.join(args.save_temp_base_dir, args.dataset)
    ckpt_path = os.path.join(ckpt_path, "vit/ViT-B_16-224.npz")
    logger.debug(f"loading model from: {ckpt_path}")
    if args.dataset == "imagenet":
        logger.debug(f"loading vit for imagenet with input shape: {args.input_shape}")
        vit_forward, params = init_vit_forward_imagenet(args, ckpt_path)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if hasattr(args, "forward"):
        assert isinstance(
            args.forward, list
        ), f"forward must be a list recieved {type(args.forward)}"
        args.forward.append(vit_forward)
        args.params.append(params)
    else:
        args.forward = [vit_forward]
        args.params = [params]


def init_vit_forward_imagenet(args, ckpt_path):
    params = checkpoint.load(ckpt_path)
    params["pre_logits"] = {}  # Need to restore empty leaf for

    model_name = "ViT-B_16"
    num_classes = 1000

    model_config = models_config.MODEL_CONFIGS[model_name]

    vit_b16 = models.VisionTransformer(num_classes=num_classes, **model_config)

    vit_b16_forward = partial(
        vit_b16.apply,
        train=False,
    )

    return vit_b16_forward, dict(params=params)
