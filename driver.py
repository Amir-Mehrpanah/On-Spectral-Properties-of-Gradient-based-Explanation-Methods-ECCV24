import argparse
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats

parser = argparse.ArgumentParser()
args = driver_helpers.base_parser(parser, configs.DefaultArgs)

num_samplers = len(args.samplers)
if args.gather_stats:
    for sindex in range(num_samplers):
        print(f"task {sindex}/{num_samplers} started.")
        start = time.time()
        stats = gather_stats(sindex, args)
        end = time.time()
        args.time_to_compute = end - start
        print(
            f"task {sindex}/{num_samplers} finsied in {args.time_to_compute:.4f}s",
            "\nnumber of samples",
            stats[args.batch_index_key] * args.batch_size,
        )

        driver_helpers.sampling_demo(args, stats)
        metadata = driver_helpers.save_stats(args, stats, sindex)
        driver_helpers.save_metadata(metadata)
else:
    raise NotImplementedError
