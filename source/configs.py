import json
import os
import sys

sys.path.append(os.getcwd())


class DefaultArgs:
    def __init__(self) -> None:
        raise NotImplementedError("This class is not meant to be instantiated")

    _args_pattern_state = {
        # "key": ["pattern", "compilation state"],
        "forward": ["i", "static"],
        "alpha_mask": ["k", "dynamic"],
        "projection": ["j", "dynamic"],
        "image": ["i", "static"],
        "baseline_mask": ["i", "dynamic"],
        "normalize_sample": ["i", "dynamic"],
        "demo": ["i", "static,meta"],
    }
    methods = ["noise_interpolation", "fisher_information"]
    architectures = ["resnet50"]
    output_layers = ["logits", "log_softmax", "softmax"]

    seed = 42
    write_demo = True
    input_shape = (1, 224, 224, 3)

    stats_log_level = 0
    monitored_statistic = "meanx2"
    output_layer = output_layers[1]  # see paper for why
    monitored_stream = "vanilla_grad_mask"
    min_change = 1e-2
    batch_size = 32
    max_batches = 10000 // batch_size
    gather_stats = True
    dataset = "imagenet"
    # args we don't want to be compiled by jax
    args_state = json.dumps({k: v[1] for k, v in _args_pattern_state.items()})
    args_pattern = json.dumps({k: v[0] for k, v in _args_pattern_state.items()})
    num_classes = 1000
    dataset_dir = "/local_storage/datasets/imagenet"
    save_raw_data_dir = "/local_storage/users/amirme/raw_data"
    save_metadata_dir = "/local_storage/users/amirme/metadata"
    jupyter_data_dir = "/local_storage/users/amirme/jupyter_data"
    visualizations_dir = os.path.join(jupyter_data_dir, "visualizations")
    profiler_dir = os.path.join(jupyter_data_dir, "profiler")

    image_height = input_shape[1]
    image_index = 0
