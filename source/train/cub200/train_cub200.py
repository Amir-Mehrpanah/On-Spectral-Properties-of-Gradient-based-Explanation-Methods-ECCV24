import tensorflow as tf

# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")

import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import flax
import flax.linen as nn
from flax.training import train_state
from flax.training import common_utils
from flax.training import checkpoints
from flax.training import lr_schedule
import optax
import orbax
import numpy as np
import dataclasses
import functools
from tqdm import tqdm
from typing import Any
import argparse
import os

import flaxmodels as fm


def cross_entropy_loss(logits, labels):
    """
    Computes the cross entropy loss.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (tensor): Cross entropy loss, shape [].
    """

    # weights = torch.ones_like(target)
    # weights[target == 1] = 1.0 / 0.04335526
    # weights[target == 2] = 1.0 / 0.59394851
    # weights[target == 3] = 1.0 / 0.31838886
    # weights[target == 4] = 1.0 / 0.04430736
    # print(weights.shape)
    # x = jax.lax.stop_gradient(x)
    # class_weights = [jnp.count_nonzero(labels == i) / float(logits.shape[0]) for i in range(logits.shape[1])]

    return (
        -jnp.sum(common_utils.onehot(labels, num_classes=logits.shape[1]) * logits)
        / labels.shape[0]
    )


def compute_metrics(logits, labels):
    """
    Computes the cross entropy loss and accuracy.

    Args:
        logits (tensor): Logits, shape [B, num_classes].
        labels (tensor): Labels, shape [B,].

    Returns:
        (dict): Dictionary containing the cross entropy loss and accuracy.
    """
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


class TrainState(train_state.TrainState):
    """
    Simple train state for the common case with a single Optax optimizer.

    Attributes:
        batch_stats (Any): Collection used to store an exponential moving
                           average of the batch statistics.
        epoch (int): Current epoch.
    """

    batch_stats: Any
    epoch: int


def restore_checkpoint(state, path):
    """
    Restores checkpoint with best validation score.

    Args:
        state (train_state.TrainState): Training state.
        path (str): Path to checkpoint.

    Returns:
        (train_state.TrainState): Training state from checkpoint.
    """
    # restore_args = flax.training.orbax_utils.restore_args_from_target(state, mesh=None)
    # return ckpt_mgr.restore(4, items=state, restore_kwargs={'restore_args': restore_args})
    return checkpoints.restore_checkpoint(path, state)


def save_checkpoint(state, step_or_metric, path):
    """
    Saves a checkpoint from the given state.

    Args:
        state (train_state.TrainState): Training state.
        step_or_metric (int of float): Current training step or metric to identify the checkpoint.
        path (str): Path to the checkpoint directory.

    """
    if jax.process_index() == 0:

        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(path, state, step_or_metric, keep=3)
        # save_args = flax.training.orbax_utils.save_args_from_target(state)
        # ckpt_mgr.save(step_or_metric, state, save_kwargs={'save_args': save_args})


def sync_batch_stats(state):
    """
    Sync the batch statistics across devices.

    Args:
        state (train_state.TrainState): Training state.

    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def configure_dataloader(ds, prerocess, num_devices, batch_size):
    # https://www.tensorflow.org/tutorials/load_data/images
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(lambda x: (prerocess(x["image"]), x["label"]), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=num_devices * batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def train_step(state, batch):

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch["image"],
            mutable=["batch_stats"],
        )

        l2_params = jax.tree_util.tree_leaves(params)
        # Computes regularization on all except batchnorm parameters.
        weight_l2 = sum(jnp.sum(x**2) for x in l2_params if x.ndim > 1)

        loss = cross_entropy_loss(logits, batch["label"]) + 1e-4 * weight_l2

        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    aux, grads = grad_fn(state.params)

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = jax.lax.pmean(grads, axis_name="batch")

    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch["label"])

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    return new_state, metrics


def eval_step(state, batch):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return compute_metrics(logits, batch["label"])


def train_and_evaluate(config):
    num_devices = jax.device_count()

    # --------------------------------------
    # Data
    # --------------------------------------
    # https://github.com/google-research/google-research/blob/master/interpretability_benchmark/utils/preprocessing_helper.py#L162
    rot = tf.keras.layers.RandomRotation(25.0 / 360.0)
    zoom = tf.keras.layers.RandomZoom([-0.2, 0.2], [-0.2, 0.2])
    translate = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
    jitter = tf.keras.layers.RandomJitter(0.2, 0.2)

    def train_prerocess(x):
        x = rot(x, training=True)
        x = zoom(x, training=True)
        x = translate(x, training=True)
        x = jitter(x, training=True)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.resize_with_crop_or_pad(x, config.img_size, config.img_size)
        
        x = tf.cast(x, dtype="float32")
        x -= config.mean_rgb
        x /= config.stddev_rgb
        return x

    def val_prerocess(x):
        x = tf.image.resize_with_crop_or_pad(x, config.img_size, config.img_size)
        x = tf.cast(x, dtype="float32")
        x -= config.mean_rgb
        x /= config.stddev_rgb
        return x

    ds_train, ds_info = tfds.load(
        name="caltech_birds2011",
        split="train",
        with_info=True,
        data_dir=config.data_dir,
    )
    ds_val = tfds.load(
        name="caltech_birds2011",
        split="validation",
        with_info=False,
        data_dir=config.data_dir,
    )

    dataset_size = ds_train.__len__().numpy()
    dataset_size_val = ds_val.__len__().numpy()
    print("Dataset size train and validation:", dataset_size, dataset_size_val)
    ds_train = configure_dataloader(
        ds_train, train_prerocess, num_devices, config.batch_size
    )
    ds_val = configure_dataloader(ds_val, val_prerocess, num_devices, config.batch_size)

    # results = tfds.benchmark(ds_train, batch_size=256, num_iter=10)
    # print("results", results)
    # assert False
    # --------------------------------------
    # Seeding, Devices, and Precision
    # --------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    else:
        dtype = jnp.float32

    platform = jax.local_devices()[0].platform

    # --------------------------------------
    # Initialize Models
    # --------------------------------------
    rng, init_rng = jax.random.split(rng)

    if config.arch == "resnet18":
        model = fm.ResNet18(
            output="log_softmax",
            pretrained=None,
            num_classes=config.num_classes,
            dtype=dtype,
            normalize=False,
        )
    elif config.arch == "resnet34":
        model = fm.ResNet34(
            output="log_softmax",
            pretrained=None,
            num_classes=config.num_classes,
            dtype=dtype,
            normalize=False,
        )
    elif config.arch == "resnet50":
        model = fm.ResNet50(
            output="log_softmax",
            pretrained=None,
            num_classes=config.num_classes,
            dtype=dtype,
            normalize=False,
        )
    elif config.arch == "resnet101":
        model = fm.ResNet101(
            output="log_softmax",
            pretrained=None,
            num_classes=config.num_classes,
            dtype=dtype,
            normalize=False,
        )
    elif config.arch == "resnet152":
        model = fm.ResNet152(
            output="log_softmax",
            pretrained=None,
            num_classes=config.num_classes,
            dtype=dtype,
            normalize=False,
        )

    variables = model.init(
        init_rng,
        jnp.ones(
            (1, config.img_size, config.img_size, config.img_channels), dtype=dtype
        ),
    )
    params, batch_stats = variables["params"], variables["batch_stats"]

    # --------------------------------------
    # Initialize Optimizer
    # --------------------------------------
    steps_per_epoch = dataset_size // config.batch_size

    """
    learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(config.learning_rate,
                                                                        steps_per_epoch,
                                                                        config.num_epochs - config.warmup_epochs,
                                                                        config.warmup_epochs)
    """
    # https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L301
    # tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay, nesterov=True)

    learning_rate_fn = optax.piecewise_constant_schedule(
        init_value=config.learning_rate,
        boundaries_and_scales={
            5 * steps_per_epoch: 1.0,
            30 * steps_per_epoch: 0.1,
            60 * steps_per_epoch: 0.01,
            80 * steps_per_epoch: 0.001,
        },
    )

    # tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=config.weight_decay, nesterov=True)

    # https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L220
    tx = optax.sgd(
        learning_rate=learning_rate_fn, momentum=config.momentum, nesterov=True
    )

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats, epoch=0
    )

    step = 0
    epoch_offset = 0

    # Checkpointing
    # TODO: Tried using this to remove warnings, but did not help. - "SaveArgs.aggregate is deprecated, please use custom TypeHandler"
    # mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    #    create=True, max_to_keep=3, keep_period=2, step_prefix='test')
    # ckpt_mgr = orbax.checkpoint.CheckpointManager(
    #    config.ckpt_dir,
    #    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)

    if config.resume:
        ckpt_path = checkpoints.latest_checkpoint(config.ckpt_dir)
        state = restore_checkpoint(state, ckpt_path)
        step = jax.device_get(state.step)
        epoch_offset = jax.device_get(state.epoch)

    state = flax.jax_utils.replicate(state)

    # --------------------------------------
    # Create train and eval steps
    # --------------------------------------
    p_train_step = jax.pmap(functools.partial(train_step), axis_name="batch")
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    # --------------------------------------
    # Training
    # --------------------------------------

    best_val_acc = 0.0
    for epoch in range(epoch_offset, config.num_epochs):
        pbar = tqdm(total=dataset_size)

        accuracy = 0.0
        n = 0
        for image, label in ds_train.as_numpy_iterator():

            pbar.update(num_devices * config.batch_size)
            image = image.astype(dtype)
            label = label.astype(dtype)

            if image.shape[0] % num_devices != 0:
                # Batch size must be divisible by the number of devices
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])

            state, metrics = p_train_step(state, {"image": image, "label": label})
            accuracy += metrics["accuracy"]
            n += 1

            if step % config.log_every == 0:
                print(f"Step: {step}")
                print("Training accuracy:", jnp.mean(accuracy))
            step += 1

        pbar.close()
        accuracy /= n

        print(f"Epoch: {epoch}")
        print("Training accuracy:", jnp.mean(accuracy))

        # --------------------------------------
        # Validation
        # --------------------------------------
        # Sync batch stats
        state = sync_batch_stats(state)

        accuracy = 0.0
        n = 0
        pbar = tqdm(total=dataset_size_val)
        for image, label in ds_val.as_numpy_iterator():
            pbar.update(num_devices * config.batch_size)

            image = image.astype(dtype)
            label = label.astype(dtype)
            if image.shape[0] % num_devices != 0:
                continue

            # Reshape images from [num_devices * batch_size, height, width, img_channels]
            # to [num_devices, batch_size, height, width, img_channels].
            # The first dimension will be mapped across devices with jax.pmap.
            image = jnp.reshape(image, (num_devices, -1) + image.shape[1:])
            label = jnp.reshape(label, (num_devices, -1) + label.shape[1:])
            metrics = p_eval_step(state, {"image": image, "label": label})
            accuracy += metrics["accuracy"]
            n += 1

        pbar.close()
        accuracy /= n
        print("Validation accuracy:", jnp.mean(accuracy))
        accuracy = jnp.mean(accuracy).item()

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            state = dataclasses.replace(
                state,
                **{
                    "step": flax.jax_utils.replicate(step),
                    "epoch": flax.jax_utils.replicate(epoch),
                },
            )
            save_checkpoint(state, accuracy, config.ckpt_dir)
