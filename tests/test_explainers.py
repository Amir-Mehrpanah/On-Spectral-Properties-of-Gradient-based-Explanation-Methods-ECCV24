from typing import Dict
import jax
import jax.numpy as jnp
import flaxmodels as fm
from PIL import Image

# functions copied from https://github.com/matthias-wright/flaxmodels/blob/main/training/resnet/training.py
def train_prerocess(x):
    x = tf.image.random_crop(x, size=(img_size, img_size, img_channels))
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_flip_left_right(x)
    # Cast to float because if the image has data type int, the following augmentations will convert it
    # to float then apply the transformations and convert it back to int.
    x = tf.cast(x, dtype='float32')
    x = tf.image.random_brightness(x, max_delta=0.5)
    x = tf.image.random_contrast(x, lower=0.1, upper=1.0)
    x = tf.image.random_hue(x, max_delta=0.5)
    x = (x - 127.5) / 127.5
    return x

def val_prerocess(x):
    x = tf.expand_dims(x, axis=0)
    #x = tf.keras.layers.experimental.preprocessing.CenterCrop(height=config.img_size, width=config.img_size)(x)
    x = tf.image.random_crop(x, size=(x.shape[0], config.img_size, config.img_size, config.img_channels))
    x = tf.squeeze(x, axis=0)
    x = tf.cast(x, dtype='float32')
    x = (x - 127.5) / 127.5
    return x

class Assets:
    
    key = jax.random.PRNGKey(0)
    model = fm.ResNet50(
        output="log_softmax",
        pretrained="imagenet",
    )
    params = resnet50.init(key, x)
    transforms = lambda img: jnp.expand_dims(
        jnp.array(img, dtype=jnp.float32) / 255.0, axis=0
    )
    # original paths
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048840.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00048864.JPEG"
    # "/local_storage/datasets/imagenet/val/n01443537/ILSVRC2012_val_00049585.JPEG"
    images: Dict[int, jax.Array] = {
        95: transforms(Image.open("tests/assets/ILSVRC2012_val_00048840.JPEG")),
        96: transforms(Image.open("tests/assets/ILSVRC2012_val_00048864.JPEG")),
        97: transforms(Image.open("tests/assets/ILSVRC2012_val_00049585.JPEG")),
    }


class TestResnet50:
    def test_grad_resnet_wrt_input(self):
        model = ResNet50(
            output="log_softmax",
            pretrained="imagenet",
        )
        x = np.random.rand(1, 224, 224, 3)
        grad = jax.grad(lambda x: model(x).sum())(x)
        assert grad.shape == x.shape
