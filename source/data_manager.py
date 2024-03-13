from datetime import datetime
import functools
import os
from typing import Any
import PIL
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import beta
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import jax.numpy as jnp
import logging

from tensorflow_datasets.core.utils import gcs_utils

from source.utils import Statistics

gcs_utils._is_gcs_disabled = True

logger = logging.getLogger(__name__)


def _bool(x) -> bool:
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False
    else:
        raise ValueError(f"expected True or False got {x}")


class TypeOrNan:
    def __init__(self, type) -> None:
        self.type = type

    def __call__(self, x: str) -> Any:
        if x.lower() == "nan" or x.lower() == "none":
            if self.type == str:
                return None
            return np.nan
        return self.type(x)


def preprocess(x, img_size, mean_rgb=None, std_rgb=None):
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=img_size,
        width=img_size,
    )(x)
    x = jnp.array(x)
    x = jnp.expand_dims(x, axis=0) / 255.0
    if mean_rgb is not None:
        x = (x - mean_rgb) / std_rgb
    return x


def save_axis(names, fig, axes, save_output_dir, dpi=300):
    plt.draw()
    for ax, name in zip(axes.flatten(), names):
        extent = (
            ax.get_tightbbox(fig.canvas.renderer)
            .transformed(fig.dpi_scale_trans.inverted())
            .padded(0.05)
        )

        path = os.path.join(save_output_dir, f"{name}.pdf")
        fig.savefig(
            path,
            bbox_inches=extent,
            transparent=True,
            dpi=dpi,
        )


# move to visualization.py
def plot_masks(masks, titles=None, imshow_args={}, ncols=5):
    ncols = ncols
    nrows = np.ceil(len(masks) / ncols).astype(int)
    scale_factor = 4
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * scale_factor, nrows * scale_factor)
    )
    for i, ax in enumerate(axes.flatten()):
        if i >= len(masks):
            break
        mask = masks.iloc[i]
        ax.imshow(mask, **imshow_args)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles.iloc[i])

    return fig, axes


def preprocess_masks(masks, preprocesses=[]):
    for process in preprocesses:
        masks = masks.apply(process)
    return masks


def preprocess_masks_ndarray(masks, preprocesses):
    for preprocess in preprocesses:
        masks = preprocess(masks)
    return masks


def aggregate_grad_mask_generic(data, agg_func, perprocess=[]):
    init_val = np.load(data.iloc[0]["grad_mask"])
    init_val = preprocess_masks_ndarray(init_val, preprocesses=perprocess)
    init_val = np.zeros_like(init_val)

    for id, row in data.iterrows():
        temp_grad = np.load(row["grad_mask"])
        temp_grad = preprocess_masks_ndarray(temp_grad, preprocesses=perprocess)
        init_val = agg_func(init_val, temp_grad, row["alpha_mask_value"])

    if isinstance(init_val, dict):
        init_val = init_val["values"]

    assert init_val.ndim == 3, f"{init_val.shape} must be 3d (H,W, 1)"
    # init_val = np.expand_dims(init_val, axis=0)
    return init_val


def running_sum(init_val, temp_grad):
    return init_val + temp_grad


def div_by_sum(grad_mask, grad_sum):
    return grad_mask / grad_sum


def unif_mul_freq(init_val, temp_grad, frequency):
    return init_val + temp_grad * (frequency ** (5 / 2))


def argmax_freq(init_val, temp_grad, frequency):
    if not isinstance(init_val, dict):
        init_val = {"values": init_val, "argmax_trace": np.zeros_like(init_val)}
    ids = init_val["argmax_trace"] < temp_grad
    init_val["values"][ids] = frequency
    init_val["argmax_trace"][ids] = temp_grad[ids]
    return init_val


def max_pixel_freq(init_val, temp_grad, frequency):
    if not isinstance(init_val, dict):
        init_val = {"values": init_val, "max_trace": np.zeros_like(init_val)}
    ids = init_val["max_trace"] < temp_grad * frequency
    init_val["values"][ids] = temp_grad[ids]
    init_val["max_trace"][ids] = temp_grad[ids] * frequency
    return init_val


def max_image_freq(init_val, temp_grad, frequency):
    if not isinstance(init_val, dict):
        init_val = {"values": init_val, "max_trace": 0}
    temp = temp_grad.sum() * frequency
    if init_val["max_trace"] < temp:
        init_val["max_trace"] = temp
        init_val["values"] = temp_grad
    return init_val


def integrated_grad(init_val, temp_grad, _):
    return init_val + temp_grad


def beta_mul_freq(init_val, temp_grad, frequency, a=2, b=2):
    return init_val + temp_grad * (frequency ** (5 / 2)) * beta.pdf(frequency, a, b)


def beta_integrated_grad(init_val, temp_grad, frequency, a=2, b=2):
    return init_val + temp_grad * beta.pdf(frequency, a, b)


def save_spectral_lens(
    data,
    save_raw_data_dir,
    agg_func=unif_mul_freq,
):
    init_val = aggregate_grad_mask_generic(data, agg_func, perprocess=[sum_channels])
    rnd = np.random.randint(0, 1000)
    path_prefix = datetime.now().strftime(f"%m%d_%H%M%S%f-{rnd}")
    save_path = os.path.join(save_raw_data_dir, f"SL_{path_prefix}.npy")
    np.save(save_path, init_val)
    return save_path


def save_arg_lens(
    data,
    save_raw_data_dir,
    agg_func=argmax_freq,
):
    init_val = aggregate_grad_mask_generic(data, agg_func, perprocess=[sum_channels])
    rnd = np.random.randint(0, 1000)
    path_prefix = datetime.now().strftime(f"%m%d_%H%M%S%f-{rnd}")
    save_path = os.path.join(save_raw_data_dir, f"AL_{path_prefix}.npy")
    np.save(save_path, init_val)
    return save_path


def save_integrated_grad(
    data,
    save_raw_data_dir,
    agg_func=integrated_grad,
    ig_elementwise=False,
    img_size=None,
    random_access_dataset=None,
    stream_statistic=None,
):
    init_val = aggregate_grad_mask_generic(data, agg_func, perprocess=[sum_channels])

    if stream_statistic == Statistics.meanx:
        # logger.debug("stream_statistic is meanx using the negative of the rankings")
        init_val = -init_val  # see the derivations in the paper

    if ig_elementwise:
        if random_access_dataset is None:
            image = PIL.Image.open(data.iloc[0]["image_path"])
            image = tf.keras.utils.img_to_array(image)
            image = preprocess(
                image,
                img_size=img_size,
            ).squeeze(0)
        else:
            image_index = data.iloc[0]["image_index"]
            image = jnp.array(random_access_dataset[int(image_index)]["image"])

        image = sum_channels(image)
        assert (
            image.shape == init_val.shape
        ), "image and grad_mask are expected to be of the same shape"
        init_val = init_val * image

    rnd = np.random.randint(0, 1000)
    path_prefix = datetime.now().strftime(f"%m%d_%H%M%S%f-{rnd}")
    save_path = os.path.join(save_raw_data_dir, f"IG_{path_prefix}.npy")
    np.save(save_path, init_val)
    return save_path


def fisher_information(dynamic_meanx2, static_meanx2, prior):
    """
    $FI = E_x[Var_y[\nabla\log y f(x)]]$ where $yf(x) = p(y|x)$
    $FI = A - B$ where
    $A = \sum_i^k E_x[(\nabla (\log f(x))_i)^2] q_i$
    $B = E_x[(\sum_i^k \nabla (\log f(x))_i q_i)^2]
    """
    static_meanx2 = static_meanx2.loc[:, "data_path"].apply(np.load)
    static_meanx2 = np.stack(static_meanx2, axis=0)

    dynamic_meanx2 = dynamic_meanx2.loc[:, "data_path"].apply(np.load)
    dynamic_meanx2 = dynamic_meanx2.to_numpy()[0]
    assert static_meanx2.shape[0] == prior.shape[0]
    e2q = (static_meanx2 * prior).sum(axis=0)
    eq2 = dynamic_meanx2
    return e2q - eq2


def minmax_normalize(x, min_=None, max_=None, return_minmax=False):
    if min_ is None:
        min_ = jnp.min(x)
    if max_ is None:
        max_ = jnp.max(x)
    x = x - min_
    x = x / (max_ - min_)
    if return_minmax:
        return x, min_, max_
    else:
        return x


def symmetric_minmax_normalize(x):
    x = x / jnp.max(jnp.abs(x))
    return x


def sum_channels(x):
    x = jnp.sum(x, axis=-1, keepdims=True)  # (H, W, C) -> (N, H, 1)
    return x


def single_query_imagenet(dataset_dir, skip, input_shape, take=1):
    args = argparse.Namespace(
        dataset_dir=dataset_dir, input_shape=input_shape, image_index=[skip, take]
    )
    query_imagenet(args)
    return args.image[0], args.label[0], args.image_path[0]


def query_imagenet(args):
    args.image = []
    args.label = []
    args.image_path = []
    image_height = args.input_shape[1]  # (N, H, W, C)
    dataset = tfds.folder_dataset.ImageFolder(root_dir=args.dataset_dir)
    dataset = dataset.as_dataset(split="val", shuffle_files=False)
    skip = args.image_index[0]
    take = args.image_index[1]
    args.image_index = []
    dataset = dataset.skip(skip)
    dataset = dataset.take(take)
    iterator = dataset.as_numpy_iterator()
    logger.info(f"dataset size is {dataset.cardinality()}")
    for i, base_stream in enumerate(iterator, skip):
        base_stream["image"] = preprocess(base_stream["image"], image_height)
        base_stream["label"] = int(base_stream["label"])
        args.image_index.append(i)
        args.image.append(base_stream["image"])
        args.label.append(base_stream["label"])
        args.image_path.append(base_stream["image/filename"].decode())

        assert args.image[-1].shape == args.input_shape


class CBIS_DDSM_CraftedDecoder(tfds.decode.Decoder):
    def __init__(self, input_shape, mean_rgb, std_rgb):
        self._input_shape = input_shape
        self._mean_rgb = mean_rgb
        self._std_rgb = std_rgb

    # just to trick tensorflow to think that this decoder is legit!
    def decode_example(self, example, feature):
        raise NotImplementedError

    def decode_example_np(self, example):
        example = tf.image.decode_jpeg(example)
        example = tf.cast(example, tf.float32)
        example = example / 255.0
        example = (example - self._mean_rgb) / self._std_rgb
        return example


class Food101CraftedDecoder(tfds.decode.Decoder):
    def __init__(self, input_shape, mean_rgb, std_rgb):
        self._input_shape = input_shape
        self._mean_rgb = mean_rgb
        self._std_rgb = std_rgb
        self.center_crop = tf.keras.layers.experimental.preprocessing.CenterCrop(
            height=self._input_shape[-2],
            width=self._input_shape[-3],
        )

    # just to trick tensorflow to think that this decoder is legit!
    def decode_example(self, example, feature):
        raise NotImplementedError

    def decode_example_np(self, example):
        example = tf.image.decode_jpeg(example)
        example = self.center_crop(example)
        example = tf.cast(example, tf.float32)
        example = example / 255.0
        example = (example - self._mean_rgb) / self._std_rgb
        return example


# to trick combination generator to think that the length of the dataset is known
class ShiftedList(list):
    def __init__(self, dataset, key, skip, take, process=None):
        self._dataset = dataset
        self._length = take
        self._key = key
        self._skip = skip
        self._process = process

    def __iter__(self):
        for i in range(self._skip, self._skip + self._length):
            if self._process is not None:
                yield self._process(self._dataset[i][self._key])
            else:
                yield self._dataset[i][self._key]

    def __getitem__(self, index):
        assert index < self._length
        if self._process is not None:
            return self._process(self._dataset[index + self._skip][self._key])
        else:
            return self._dataset[index + self._skip][self._key]

    def __len__(self):
        return self._length

    def __repr__(self) -> str:
        return f"shifted list of {self._length} items from {self._skip} to {self._skip + self._length - 1} with process {self._process}"


def single_query_food101(dataset_dir, skip, input_shape, take=1):
    preprocess_mean_rgb = np.array([0.561, 0.440, 0.312])
    preprocess_std_rgb = np.array([0.252, 0.256, 0.259])
    args = argparse.Namespace(
        dataset_dir=dataset_dir,
        input_shape=input_shape,
        image_index=[skip, take],
        mean_rgb=preprocess_mean_rgb,
        std_rgb=preprocess_std_rgb,
    )
    query_food101(args)
    return args.image[0], args.label[0], args.image_path[0]


def query_food101(args):
    args.image = []
    args.label = []
    args.image_path = []
    food_dataset = tfds.data_source(
        "food101",
        split="validation",
        data_dir=args.dataset_dir,
        download=False,
        decoders={
            "image": Food101CraftedDecoder(
                args.input_shape,
                args.mean_rgb,
                args.std_rgb,
            )
        },
    )

    skip = args.image_index[0]
    take = args.image_index[1]
    args.image = ShiftedList(food_dataset, "image", skip, take, jnp.array)
    args.label = ShiftedList(food_dataset, "label", skip, take, jnp.array)
    args.image_index = list(range(skip, take + skip))
    args.image_path = ["NA"] * len(args.image_index)


def single_query_curated_breast_imaging_ddsm(dataset_dir, skip, input_shape, take=1):
    preprocess_mean_rgb = np.array([0.359])
    preprocess_std_rgb = np.array([1.0])
    args = argparse.Namespace(
        dataset_dir=dataset_dir,
        input_shape=input_shape,
        image_index=[skip, take],
        mean_rgb=preprocess_mean_rgb,
        std_rgb=preprocess_std_rgb,
    )
    query_curated_breast_imaging_ddsm(args)
    return args.image[0], args.label[0], args.image_path[0]


def query_curated_breast_imaging_ddsm(args):
    args.image = []
    args.label = []
    args.image_path = []
    cbis_ddsm_dataset = tfds.data_source(
        "curated_breast_imaging_ddsm",
        split="validation",
        data_dir=args.dataset_dir,
        download=False,
        decoders={
            "image": CBIS_DDSM_CraftedDecoder(
                args.input_shape,
                args.mean_rgb,
                args.std_rgb,
            )
        },
    )

    skip = args.image_index[0]
    take = args.image_index[1]
    args.image = ShiftedList(cbis_ddsm_dataset, "image", skip, take, jnp.array)
    args.label = ShiftedList(cbis_ddsm_dataset, "label", skip, take, jnp.array)
    args.image_path = ShiftedList(cbis_ddsm_dataset, "id", skip, take)
    args.image_index = list(range(skip, take + skip))


def npy_header_offset(npy_path):
    with open(str(npy_path), "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("Invalid NPY file.")
        version_major, version_minor = f.read(2)
        if version_major == 1:
            header_len_size = 2
        elif version_major == 2:
            header_len_size = 4
        else:
            raise ValueError(
                "Unknown NPY file version {}.{}.".format(version_major, version_minor)
            )
        header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
        header = f.read(header_len)
        if not header.endswith(b"\n"):
            raise ValueError("Invalid NPY file.")
        return f.tell()


def _tf_parse_image_fn_imagenet(image_path, input_shape):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.keras.layers.experimental.preprocessing.CenterCrop(
        height=input_shape[-2],
        width=input_shape[-3],
    )(image)
    return image


def _blur_baseline(image):
    image = tf.expand_dims(image, axis=0)
    image = tf.nn.avg_pool2d(
        image,
        ksize=100,
        strides=1,
        padding="SAME",
        data_format="NHWC",
    )
    image = tf.squeeze(image, axis=0)
    return image


def _black_baseline(image):
    return tf.zeros_like(image)


def _masking_q(image, baseline, explanation, label, q, direction, verbose=False):
    explanation_q = tfp.stats.percentile(
        explanation,
        100 - q,
        axis=(0, 1),
        keepdims=True,
    )

    if direction == "deletion":
        explanation_q = explanation <= explanation_q
    else:
        explanation_q = explanation >= explanation_q

    explanation_q = tf.cast(explanation_q, tf.float32)
    masked_image = image * explanation_q + baseline * (1 - explanation_q)
    if verbose:
        return {
            "original_image": image,
            "saliency": explanation,
            "masked_image": masked_image,
            "baseline": baseline,
            "label": label,
            "actual_q": tf.reduce_mean(explanation_q),
        }

    return {
        "masked_image": masked_image,
        "label": label,
        "actual_q": tf.reduce_mean(explanation_q),
    }


def curated_breast_imaging_ddsm_loader_from_metadata(
    sl_metadata,
    q,
    direction,
    input_shape,
    baseline="blur",
    batch_size=128,
    prefetch_factor=4,
    verbose=False,
    dataset_dir=None,
    mean_rgb=np.array([0.359]),
    std_rgb=np.array([1.0]),
):
    logger.info(
        f"creating dataloader... the dataset shape for loader is {sl_metadata.shape}"
    )
    cbis_ddsm_dataset = tfds.data_source(
        "curated_breast_imaging_ddsm",
        split="validation",
        data_dir=dataset_dir,
        download=False,
        decoders={
            "image": CBIS_DDSM_CraftedDecoder(
                input_shape,
                mean_rgb,
                std_rgb,
            )
        },
    )

    _masking_q_fn = functools.partial(
        _masking_q,
        q=q,
        direction=direction,
        verbose=verbose,
    )

    def _cbis_ddsm_generator():
        for index, image_index in enumerate(sl_metadata["image_index"].values):
            image_index = int(image_index)
            sample = cbis_ddsm_dataset[image_index]
            sample["explanation"] = np.load(sl_metadata["data_path"].values[index])

            if sample["explanation"].shape[-1] != 1:
                sample["explanation"] = sample["explanation"].sum(
                    axis=-1, keepdims=True
                )

            if baseline == "blur":
                sample["baseline"] = _blur_baseline(sample["image"])
                yield _masking_q_fn(**sample)
            elif baseline == "black":
                sample["baseline"] = _black_baseline(sample["image"])
                yield _masking_q_fn(**sample)
            else:
                raise ValueError(f"baseline {baseline} is not supported")

    logger.debug(f"creating the slq_dataset from generators.")
    slq_dataset = tf.data.Dataset.from_generator(
        _cbis_ddsm_generator,
        output_signature={
            "masked_image": tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
            "actual_q": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )

    logger.info(
        f"dataloader value of q is set to {q}, batch_size is {batch_size}, prefetch_factor is {prefetch_factor}, verbose is {verbose}"
    )
    logger.debug(f"batching and prefetching the slq_dataset.")
    slq_dataset = slq_dataset.batch(batch_size)
    slq_dataset = slq_dataset.prefetch(prefetch_factor)
    return slq_dataset


def food101_loader_from_metadata(
    sl_metadata,
    q,
    direction,
    input_shape,
    baseline="blur",
    batch_size=128,
    prefetch_factor=4,
    verbose=False,
    dataset_dir=None,
    mean_rgb=np.array([0.561, 0.440, 0.312]),
    std_rgb=np.array([0.252, 0.256, 0.259]),
):
    logger.info(
        f"creating dataloader... the dataset shape for loader is {sl_metadata.shape}"
    )
    food_dataset = tfds.data_source(
        "food101",
        split="validation",
        data_dir=dataset_dir,
        download=False,
        decoders={
            "image": Food101CraftedDecoder(
                input_shape,
                mean_rgb,
                std_rgb,
            )
        },
    )

    _masking_q_fn = functools.partial(
        _masking_q,
        q=q,
        direction=direction,
        verbose=verbose,
    )

    def _food101_generator():
        for index, image_index in enumerate(sl_metadata["image_index"].values):
            image_index = int(image_index)
            sample = food_dataset[image_index]
            sample["explanation"] = np.load(sl_metadata["data_path"].values[index])

            if sample["explanation"].shape[-1] != 1:
                sample["explanation"] = sample["explanation"].sum(
                    axis=-1, keepdims=True
                )

            if baseline == "blur":
                sample["baseline"] = _blur_baseline(sample["image"])
                yield _masking_q_fn(**sample)
            elif baseline == "black":
                sample["baseline"] = _black_baseline(sample["image"])
                yield _masking_q_fn(**sample)
            else:
                raise ValueError(f"baseline {baseline} is not supported")

    logger.debug(f"creating the slq_dataset from generators.")
    slq_dataset = tf.data.Dataset.from_generator(
        _food101_generator,
        output_signature={
            "masked_image": tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
            "actual_q": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )

    logger.info(
        f"dataloader value of q is set to {q}, batch_size is {batch_size}, prefetch_factor is {prefetch_factor}, verbose is {verbose}"
    )
    logger.debug(f"batching and prefetching the slq_dataset.")
    slq_dataset = slq_dataset.batch(batch_size)
    slq_dataset = slq_dataset.prefetch(prefetch_factor)
    return slq_dataset


def imagenet_loader_from_metadata(
    sl_metadata,
    q,
    direction,
    input_shape,
    baseline="blur",
    batch_size=128,
    prefetch_factor=4,
    verbose=False,
):
    logger.info(
        f"creating dataloader... the dataset shape for loader is {sl_metadata.shape}"
    )
    header_offset = npy_header_offset(sl_metadata["data_path"].values[0])
    shape_size = np.prod(input_shape) * tf.float32.size
    logger.debug(
        f"header offset is {header_offset}, input_shape is {input_shape} with size {shape_size}"
    )

    explanation_dataset = tf.data.FixedLengthRecordDataset(
        sl_metadata["data_path"].values, shape_size, header_bytes=header_offset
    )
    explanation_dataset = explanation_dataset.map(
        lambda s: tf.reshape(tf.io.decode_raw(s, tf.float32), input_shape),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    image_dataset = tf.data.Dataset.from_tensor_slices(sl_metadata["image_path"].values)
    _parse_fn = functools.partial(_tf_parse_image_fn_imagenet, input_shape=input_shape)
    image_dataset = image_dataset.map(
        _parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if baseline == "blur":
        logger.debug("creating blurred image dataloader for baseline")
        baseline_dataset = image_dataset.map(
            _blur_baseline,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        logger.debug("creating black image dataloader for baseline")
        baseline_dataset = image_dataset.map(
            _black_baseline,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    label_dataset = tf.data.Dataset.from_tensor_slices(sl_metadata["label"].values)

    slq_dataset = tf.data.Dataset.zip(
        (
            image_dataset,
            baseline_dataset,
            explanation_dataset,
            label_dataset,
        )
    )

    logger.info(
        f"dataloader value of q is set to {q}, batch_size is {batch_size}, prefetch_factor is {prefetch_factor}, verbose is {verbose}"
    )
    _masking_q_fn = functools.partial(
        _masking_q,
        q=q,
        direction=direction,
        verbose=verbose,
    )
    slq_dataset = slq_dataset.map(
        _masking_q_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    slq_dataset = slq_dataset.batch(batch_size)
    slq_dataset = slq_dataset.prefetch(prefetch_factor)

    return slq_dataset


def query_cifar10(args):
    args.image = []
    args.label = []
    args.image_path = []
    image_height = args.input_shape[1]  # (N, H, W, C)
    dataset = tfds.load(
        "cifar10",
        split="test",
        shuffle_files=False,
        data_dir=args.dataset_dir,
        download=False,
    )
    skip = args.image_index[0]
    take = args.image_index[1]
    args.image_index = []
    dataset = dataset.skip(skip)
    dataset = dataset.take(take)
    iterator = dataset.as_numpy_iterator()
    logger.info(f"dataset size is {dataset.cardinality()}")
    for i, base_stream in enumerate(iterator):
        base_stream["image"] = preprocess(base_stream["image"], image_height)
        args.image.append(base_stream["image"])
        args.label.append(base_stream["label"])
        args.image_path.append(base_stream["id"].decode())
        args.image_index.append(i)
        logger.info(f"image path is {args.image_path[-1]}")
        assert args.image[-1].shape == args.input_shape, (
            f"image shape is {args.image[-1].shape}, "
            f"expected input shape is {args.input_shape}"
        )
