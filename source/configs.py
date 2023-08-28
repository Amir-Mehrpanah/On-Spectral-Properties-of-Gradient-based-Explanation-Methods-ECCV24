import os
import sys

sys.path.append(os.getcwd())


class DefaultArgs:
    def __init__(self) -> None:
        raise NotImplementedError("This class is not meant to be instantiated")

    methods = ["noise_interpolation", "fisher_information"]
    architecures = ["resnet50"]

    dry_run = True
    seed = 0 if dry_run else 42
    method = methods[0] if dry_run else None

    architecure = architecures[0] if dry_run else None
    input_shape = (1, 224, 224, 3)

    monitored_statistic = "meanx2"
    monitored_stream = "vanilla_grad_mask"
    max_batches = 1 if dry_run else 10000
    min_change = 1e-6
    batch_size = 2 if dry_run else 32

    dataset = "imagenet"
    num_classes = 1000
    dataset_dir = "/local_storage/datasets/imagenet"
    dataset_dir = dataset_dir if not dry_run else "tests/assets"

    save_raw_data_dir = "/local_storage/users/amirme/raw_data"
    save_raw_data_dir = save_raw_data_dir if not dry_run else "outputs/raw_data"
    save_metadata_dir = "/local_storage/users/amirme/metadata"
    save_metadata_dir = save_metadata_dir if not dry_run else "outputs/metadata"

    tensorboard_dir = "/local_storage/users/amirme/tensorboard_logs/"
    tensorboard_dir = tensorboard_dir if not dry_run else "outputs/tensorboard_logs"

    visualizations_dir = os.path.join(tensorboard_dir, "visualizations")
    profiler_dir = os.path.join(tensorboard_dir, "profiler")

    image_height = input_shape[1]
    image_index = 0
    dataset_skip_index = 0
