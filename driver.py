import argparse
import logging
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats
from source.inconsistency_measures import measure_inconsistency
from source import project_manager
from source.utils import Action


parser = argparse.ArgumentParser()
driver_args, action_args = driver_helpers.base_parser(parser, configs.DefaultArgs)

logger = logging.getLogger(__name__)
logger.setLevel(logger.getEffectiveLevel())


if driver_args.action == Action.gather_stats:
    iterator = enumerate(action_args.samplers_and_kwargs)
    for sindex, (sampler, static_kwargs, dynamic_kwargs, meta_kwargs) in iterator:
        logger.info(f"task {sindex}/{action_args.num_samplers} started.")
        stats, stats_metadata = gather_stats(sampler, dynamic_kwargs, meta_kwargs)
        logger.info(
            f"task {sindex}/{action_args.num_samplers} "
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
        saving_metadata = driver_helpers.save_gather_stats_data(
            driver_args.save_raw_data_dir,
            driver_args.skip_data,
            stats,
        )
        driver_helpers.save_gather_stats_metadata(
            driver_args.save_metadata_dir,
            # driver_args.skip_data,
            {
                **stats_metadata,  # stats dependent metadata
                **saving_metadata,  # raw data dependent metadata
                **meta_kwargs,  # stats independent metadata
            },
        )
elif driver_args.action == Action.merge_stats:
    project_manager.merge_experiment_metadata(
        driver_args.save_metadata_dir,
    )
elif driver_args.action == Action.compute_inconsistency:
    stats = measure_inconsistency(
        action_args.data_loader,
        action_args.inconsistency_measure,
    )
    driver_helpers.save_inconsistency(
        driver_args.save_metadata_dir,
        stats,  # raw data dependent metadata
        action_args.pivot_column,
        action_args.inconsistency_measure_name,
    )
else:
    raise NotImplementedError
