import argparse
import time
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
args.ttc = end - start

print(f"sampling finsied in {args.ttc}s")
print("number of samples", args.stats[args.batch_index_key] * args.batch_size)

name_prefix, npy_file_paths = driver_helpers.save_stats(args, stats)
driver_helpers.save_metadata(args, name_prefix)
