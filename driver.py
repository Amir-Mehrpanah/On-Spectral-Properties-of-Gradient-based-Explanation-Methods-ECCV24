import argparse
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats, compute_stats

parser = argparse.ArgumentParser()
args = driver_helpers.base_parser(parser, configs.DefaultArgs)

start = time.time()
if args.gather_stats:
    print("sampling started")
    args.stats = gather_stats(args)
elif args.compute_stats:
    print("loading stats")
    args.stats = compute_stats(args)
end = time.time()
args.time_to_compute = end - start
print(f"task finsied in {args.time_to_compute:.4f}s")
print("number of samples", args.stats[args.batch_index_key] * args.batch_size)

driver_helpers.sampling_demo(args)
driver_helpers.inplace_save_stats(args)
driver_helpers.inplace_save_metadata(args)
