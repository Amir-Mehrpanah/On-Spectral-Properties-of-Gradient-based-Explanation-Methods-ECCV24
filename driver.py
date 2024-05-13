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

driver_helpers.gpu_preallocation()

parser = argparse.ArgumentParser()
driver_args, action_args = driver_helpers.base_parser(parser, configs.DefaultArgs)

logger = logging.getLogger(__name__)
logger.setLevel(logger.getEffectiveLevel())


if driver_args.action == Action.gather_stats:
    iterator = enumerate(action_args.samplers_and_kwargs)
    for sindex, (sampler, static_kwargs, dynamic_kwargs, meta_kwargs) in iterator:
        logger.info(f"task {sindex}/{action_args.num_samplers} started.")
        logger.debug(f"static_kwargs: {static_kwargs.keys()}")
        logger.debug(f"static_kwargs: {dynamic_kwargs.keys()}")
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
elif driver_args.action == Action.compute_entropy:
    project_manager.compute_entropy(
        driver_args.save_metadata_dir,
    )
elif driver_args.action == Action.merge_stats:
    project_manager.merge_experiment_metadata(
        driver_args.save_metadata_dir,
        action_args.glob_path,
        action_args.file_name,
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
elif driver_args.action == Action.compute_integrated_grad:
    project_manager.compute_integrated_grad(
        driver_args.save_metadata_dir,
        driver_args.save_raw_data_dir,
        input_shape=action_args.input_shape,
        alpha_mask_name=action_args.alpha_mask_name,
        alpha_prior=action_args.alpha_prior,
        projection_type=action_args.projection_type,
        projection_top_k=action_args.projection_top_k,
        random_access_dataset=action_args.random_access_dataset,
    )
elif driver_args.action == Action.compute_accuracy_at_q:
    driver_helpers.compute_accuracy_at_q(
        driver_args.save_metadata_dir,
        action_args.sl_metadata,
        action_args.save_file_name_prefix,
        action_args.q,
        action_args.q_direction,
        action_args.q_baseline_mask,
        action_args.forward,
        action_args.params,
        action_args.slq_dataloader,
    )
else:
    raise NotImplementedError
