import json
import logging
import os
import sys

sys.path.append(os.getcwd())
from source.utils import Action, InconsistencyMeasures, Statistics, StreamNames


class DefaultArgs:
    def __init__(self) -> None:
        raise NotImplementedError("This class is not meant to be instantiated")

    _args_pattern_state = {
        # "key": ["pattern", "compilation state"],
        # default behavior is to compile all args (all static)
        "alpha_mask": ["j", "dynamic"],
        "projection": ["i", "static"],
    }
    inconsistency_measures = [v for v in dir(InconsistencyMeasures) if "__" not in v]
    methods = ["noise_interpolation", "fisher_information"]
    logging_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
    architectures = ["resnet50", "resnet50-randomized", "resnet18"]
    output_layers = ["logits", "log_softmax", "softmax"]
    actions = [v for v in dir(Action) if "__" not in v]

    c1 = 0.01**2  # SSIM constant
    c2 = 0.03**2  # SSIM constant
    q_direction = "deletion"
    downsampling_factor = 5
    prefetch_factor = 4
    pivot_indices = ["image_index", "projection_index"]
    pivot_column = "alpha_mask_value"
    seed = 42
    write_demo = True
    input_shape = (1, 224, 224, 3)
    logging_level = logging.INFO
    stats_log_level = 0
    skip_data = None
    monitored_statistic = Statistics.meanx2
    output_layer = output_layers[1]  # see paper for why
    monitored_stream = StreamNames.vanilla_grad_mask
    min_change = 1e-2
    batch_size = 32
    max_batches = 10000 // batch_size
    action = Action.gather_stats
    dataset = "no_default_dataset"
    # args we don't want to be compiled by jax
    args_state = json.dumps(
        {k: v[1] for k, v in _args_pattern_state.items()}, separators=(";", ":")
    )
    args_pattern = json.dumps(
        {k: v[0] for k, v in _args_pattern_state.items()}, separators=(";", ":")
    )
    num_classes = 1000
    dataset_dir = "no_default_path"
    save_raw_data_dir = "no_default_path"
    save_metadata_dir = "no_default_path"
    jupyter_data_dir = "no_default_path"
    save_temp_base_dir = "no_default_path"
    visualizations_dir = os.path.join(jupyter_data_dir, "visualizations")
    profiler_dir = os.path.join(jupyter_data_dir, "profiler")

    image_height = input_shape[1]
    image_index = 0
