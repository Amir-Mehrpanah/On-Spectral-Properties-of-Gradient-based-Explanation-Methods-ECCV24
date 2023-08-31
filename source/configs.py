import os
import sys

sys.path.append(os.getcwd())


class DefaultArgs:
    def __init__(self) -> None:
        raise NotImplementedError("This class is not meant to be instantiated")

    methods = ["noise_interpolation", "fisher_information"]
    architectures = ["resnet50"]

    seed = 42
    write_demo = True
    input_shape = (1, 224, 224, 3)

    monitored_statistic = "meanx2"
    monitored_stream = "vanilla_grad_mask"
    min_change = 1e-2
    batch_size = 32
    max_batches = 10000 // batch_size

    dataset = "imagenet"
    num_classes = 1000

    dataset_dir = "/local_storage/datasets/imagenet"
    save_raw_data_dir = "/local_storage/users/amirme/raw_data"
    save_metadata_dir = "/local_storage/users/amirme/metadata"
    jupyter_data_dir = "/local_storage/users/amirme/jupyter_data"
    visualizations_dir = os.path.join(jupyter_data_dir, "visualizations")
    profiler_dir = os.path.join(jupyter_data_dir, "profiler")

    image_height = input_shape[1]
    image_index = 0
