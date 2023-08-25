import argparse
import sys
import numpy as np
import time
from source import helpers, configs
from source.helpers import gather_stats


parser = argparse.ArgumentParser()
args = helpers.base_parser(parser, configs.DefaultArgs)

start = time.time()
time.sleep(1.1)
# stats = gather_stats(
#     args.seed,
#     args.abstract_process,
#     args.batch_size,
#     args.max_batches,
#     args.min_change,
#     args.stats,
#     args.monitored_statistic_source_key,
#     args.monitored_statistic_key,
#     args.batch_index_key,
# )
end = time.time()
print(f"Time: {end - start}s")
print("number of samples", args.stats[args.batch_index_key] * args.batch_size)

np.savez(args.save_path, **args.stats)

