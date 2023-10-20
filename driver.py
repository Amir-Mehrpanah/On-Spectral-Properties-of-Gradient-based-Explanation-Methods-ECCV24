import argparse
import logging
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
driver_args, method_args = driver_helpers.base_parser(parser, configs.DefaultArgs)

if driver_args.action == "gather_stats":
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
            f"task {sindex}/{num_samplers}"
            f"finsied in {stats_metadata['time_to_compute']:.3f}s"
            "\nnumber of samples"
            f"{stats_metadata['batch_index'] * meta_kwargs['batch_size']}",
        )
        if driver_args.write_demo:
            driver_helpers.sample_demo(
                static_kwargs, dynamic_kwargs, meta_kwargs, stats
            )
        saved_metadata = driver_helpers.save_stats(driver_args.save_raw_data_dir, stats)
        driver_helpers.save_metadata(
            driver_args.save_metadata_dir,
            {
                **stats_metadata,  # stats dependent metadata
                **saved_metadata,  # raw data dependent metadata
                **meta_kwargs,  # stats independent metadata
            },
        )
else:
    raise NotImplementedError
