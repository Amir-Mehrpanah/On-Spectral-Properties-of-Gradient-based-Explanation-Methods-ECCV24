import os
import sys

sys.path.append(os.getcwd())


class DefaultArgs:
    def __init__(self) -> None:
        raise NotImplementedError("This class is not meant to be instantiated")

    methods = ["noise_interpolation", "fisher_information"]
    forward = "resnet50"
    monitored_statistic = "meanx2"
    monitored_stream = "vanilla_grad_mask"
    method = methods[0]
    max_batches = 10000
    image_index = 0
    min_change = 1e-6
    seed = 0
    batch_size = 32
    num_classes = 1000
    input_shape = (1, 224, 224, 3)

    dataset = "imagenet"
    dataset_dir = "/local_storage/datasets/imagenet"

    save_raw_data_dir = "/local_storage/users/amirme/raw_data"
    save_metadata_dir = "/local_storage/users/amirme/metadata"

    tensorboard_dir = "/local_storage/users/amirme/tensorboard_logs/"
    visualizations_dir = os.path.join(tensorboard_dir, "visualizations")
    profiler_dir = os.path.join(tensorboard_dir, "profiler")

    image_height = input_shape[1]
    dataset_skip_index = 0
