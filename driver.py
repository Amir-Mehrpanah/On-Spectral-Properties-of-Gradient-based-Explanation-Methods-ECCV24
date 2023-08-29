import argparse
import time
import os
import sys

sys.path.append(os.getcwd())
from source import driver_helpers, configs
from source.operations import gather_stats

parser = argparse.ArgumentParser()
args = driver_helpers.base_parser(parser, configs.DefaultArgs)

start = time.time()
print("sampling started")
time.sleep(0.1)
stats = gather_stats(
    args.seed,
    args.abstract_process,
    args.batch_size,
    args.max_batches,
    args.min_change,
    args.stats,
    args.monitored_statistic_source_key,
    args.monitored_statistic_key,
    args.batch_index_key,
)
end = time.time()
args.time_to_compute = end - start

print(f"sampling finsied in {args.time_to_compute:.4f}s")
print("number of samples", stats[args.batch_index_key] * args.batch_size)

driver_helpers.inplace_save_stats(args, stats)
driver_helpers.inplace_save_metadata(args)
