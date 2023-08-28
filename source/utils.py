from collections import namedtuple
from functools import partial, update_wrapper

class AbstractFunction:
    def __init__(self, func) -> None:
        self.func = func
        self.params = {}
        update_wrapper(self, func)

    def __call__(self, **kwargs):
        self.params.update(kwargs)
        return self

    def concretize(self):
        return partial(self.func, **self.params)


class StreamNames:
    batch_index = 10  # does it affect performance if we use a string instead of an int?
    vanilla_grad_mask = 11
    results_at_projection = 12
    log_probs = 13


class Statistics:
    none = 0
    meanx = 1
    meanx2 = 2
    abs_delta = 3


Stream = namedtuple("Stream", ["name", "statistic"])
