import argparse
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats

parser = argparse.ArgumentParser()
args = driver_helpers.base_parser(parser, configs.DefaultArgs)

if args.gather_stats:
    for sampler in args.samplers:
        start = time.time()
        print("sampling started")
        stats = gather_stats(sampler, args)
        end = time.time()
        args.time_to_compute = end - start
        print(f"task finsied in {args.time_to_compute:.4f}s")
        print(
            "number of samples",
            stats[args.batch_index_key] * args.batch_size,
        )

        driver_helpers.sampling_demo(args)
        driver_helpers.inplace_save_stats(args)
        driver_helpers.inplace_save_metadata(args)
else:
    raise NotImplementedError
