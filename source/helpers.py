from collections import namedtuple


class StreamNames:
    batch_index = 10
    vanilla_grad_mask = 11
    results_at_projection = 12
    log_probs = 13


class Statistics:
    none = 0
    meanx = 1
    meanx2 = 2
    abs_delta = 3


Stream = namedtuple("Stream", ["name", "statistic"])
