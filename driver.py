import argparse
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats

parser = argparse.ArgumentParser()
args = driver_helpers.base_parser(parser, configs.DefaultArgs)

for arg_pattern in driver_helpers.iterate_pattern_sampler_args(args):
    start = time.time()
    if arg_pattern.gather_stats:
        print("sampling started")
        arg_pattern.stats = gather_stats(arg_pattern)
    else:
        raise NotImplementedError
    end = time.time()
    arg_pattern.time_to_compute = end - start
    print(f"task finsied in {arg_pattern.time_to_compute:.4f}s")
    print(
        "number of samples",
        arg_pattern.stats[arg_pattern.batch_index_key] * arg_pattern.batch_size,
    )

    driver_helpers.sampling_demo(arg_pattern)
    driver_helpers.inplace_save_stats(arg_pattern)
    driver_helpers.inplace_save_metadata(arg_pattern)
