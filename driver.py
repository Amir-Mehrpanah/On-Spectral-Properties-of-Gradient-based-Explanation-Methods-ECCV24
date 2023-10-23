import argparse
import logging
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats, measure_consistency
from source import project_manager
from source.utils import Action

logger = logging.getLogger(__name__)
logger.setLevel(logger.getEffectiveLevel())

parser = argparse.ArgumentParser()
driver_args, method_args = driver_helpers.base_parser(parser, configs.DefaultArgs)


if driver_args.action == Action.gather_stats:
    num_samplers = len(method_args.samplers)
    iterator = zip(
        method_args.samplers,
        method_args.static_kwargs,
        method_args.dynamic_kwargs,
        method_args.meta_kwargs,
    )
    for sindex, (sampler, static_kwargs, dynamic_kwargs, meta_kwargs) in enumerate(
        iterator
    ):
        logger.info(f"task {sindex}/{num_samplers} started.")
        stats, stats_metadata = gather_stats(sampler, dynamic_kwargs, meta_kwargs)
        logger.info(
            f"task {sindex}/{num_samplers} "
            f"finsied in {stats_metadata['time_to_compute']:.3f}s "
            "\nnumber of samples "
            f"{stats_metadata['batch_index'] * meta_kwargs['batch_size']}",
        )
        if driver_args.write_demo:
            driver_helpers.sample_demo(
                static_kwargs,
                dynamic_kwargs,
                meta_kwargs,
                stats,
            )
        saving_metadata = driver_helpers.save_stats(
            driver_args.save_raw_data_dir,
            stats,
            driver_args.path_prefix,
        )
        driver_helpers.save_metadata(
            driver_args.save_metadata_dir,
            {
                **stats_metadata,  # stats dependent metadata
                **saving_metadata,  # raw data dependent metadata
                **meta_kwargs,  # stats independent metadata
            },
        )
        project_manager.merge_experiment_metadata(
            driver_args.save_metadata_dir,
            driver_args.path_prefix,
        )
elif driver_args.action == Action.compute_consistency:
    data_loader = project_manager.alpha_group_loader(  # todo move this to driver_helpers._parse_measure_consistency_args
        driver_args.save_metadata_dir,
        driver_args.batch_size,
        driver_args.path_prefix,
    )
    stats = measure_consistency(data_loader)
    driver_helpers.save_metadata(
        driver_args.save_metadata_dir,
        stats,  # raw data dependent metadata
    )
else:
    raise NotImplementedError
