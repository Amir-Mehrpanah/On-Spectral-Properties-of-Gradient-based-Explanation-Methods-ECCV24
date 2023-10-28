from collections import namedtuple
import inspect
from collections import OrderedDict
import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Action:
    gather_stats = "gather_stats"
    compute_consistency = "compute_consistency"
    merge_stats = "merge_stats"


class ConsistencyMeasures:
    cosine_distance = "cosine_distance"
    dssim = "dssim"


def debug_nice(x, r=0, max_depth=1):
    if inspect.isfunction(x):
        return f"{x.__name__}"
    if isinstance(x, np.ndarray):
        return f"np.ndarray of shape {x.shape}"
    if isinstance(x, list):
        if r > max_depth:
            return f"list of length {len(x)}"
        nice = [debug_nice(v, r=r + 1) for v in x]
        return f"list {nice}"
    if isinstance(x, dict):
        if r > max_depth:
            return f"dict of length {len(x)}"
        nice = {k: debug_nice(v, r=r + 1) for k, v in x.items()}
        return f"dict {nice}"
    if isinstance(x, tuple):
        if r > max_depth:
            return f"tuple of length {len(x)}"
        nice = tuple(debug_nice(v, r=r + 1) for v in x)
        return f"tuple {nice}"
    return f"{x}"


class AbstractFunction:
    __cache = {}

    class NoArg:
        pass

    def __init__(self, func) -> None:
        self.func = func
        params = inspect.signature(func).parameters.keys()
        self.params = OrderedDict({k: self.NoArg for k in params})

    def __call__(self, **kwargs):
        for k in kwargs:
            assert k in self.params, f'partial input "{k}" is unknown'
        self.params.update(**kwargs)
        return self

    def __repr__(self) -> str:
        return f"abstract {self.func.__name__}(static_args={list(self.params.keys())})"

    def concretize(self):
        hash_args = tuple(id(arg) for arg in self.params.values())
        if logger.isEnabledFor(logging.DEBUG):
            nice_params = debug_nice(self.params)
            logger.debug(
                f"concretizing {self.func} with static arguments {nice_params}"
            )
        if hash_args in self.__cache:
            logger.warning(
                "concretization returned a cached abstact function for identical signature in partial calls",
            )
            return self.__cache[hash_args]

        def concrete_func(*args):
            i = 0
            temp_params = self.params.copy()
            if logger.isEnabledFor(logging.DEBUG):
                nice_params = debug_nice(temp_params)
                nice_pos_args = debug_nice(args)
                logger.debug(
                    f"concrete function {self.func} called with static arguments {nice_params} and dynamic arguments {nice_pos_args}"
                )
            for key, param in temp_params.items():
                if param is self.NoArg:
                    temp_params[key] = args[i]
                    i += 1
            assert i == len(
                args
            ), f"number of positional arguments does not match the concrete function when calling {self.func} with {self.params} and positional arguments {args}"
            return self.func(**temp_params)

        self.__cache[hash_args] = concrete_func
        return concrete_func


def combine_patterns(pattern, values):
    """
    a pattern is not sensitive to actual values, but to the unique keys e.g.
    {"a": "b", "c": "d"} == {"a": "e", "c": "f"}
    {"a": "b", "c": "b"} == {"a": "f", "c": "f"}
    {"a": "b", "c": "b"} != {"a": "b", "c": "f"}
    returns a generator of values for each item in the list according to the pattern
    e.g.
    >>> pattern = {"a": "i", "c": "j"}
    >>> values = {"a": [1, 2], "c": [3, 4]}
    >>> list(combine_patterns(pattern, values))
    [
        {"a": 1, "c": 3},
        {"a": 1, "c": 4},
        {"a": 2, "c": 3},
        {"a": 2, "c": 4},
    ]
    """
    pattern_keys = list(pattern.keys())
    pattern_values = list(pattern.values())
    list_values = [values[k] for k in pattern_keys]
    unique_pattern_values = list(set(pattern_values))
    comb_index = [pattern_values.index(v) for v in unique_pattern_values]
    len_values = [len(list_values[i]) for i in comb_index]
    range_values = [range(l) for l in len_values]
    combinations = list(itertools.product(*range_values))
    pattern_index = [unique_pattern_values.index(v) for v in pattern_values]
    results = []
    for combination in combinations:
        value_indices = [combination[index] for index in pattern_index]
        temp_values = {k: values[k][i] for i, k in zip(value_indices, pattern_keys)}
        results.append(temp_values)
    return results


class StreamNames:
    batch_index = "index"
    vanilla_grad_mask = "vanilla_grad_mask"
    results_at_projection = "results_at_projection"
    log_probs = "log_probs"
    image = "image"


class Statistics:
    none = "none"
    meanx = "meanx"
    meanx2 = "meanx2"
    abs_delta = "abs_delta"


Stream = namedtuple("Stream", ["name", "statistic"])


class Switch:
    def __init__(self):
        self.key_values = {}

    def register(self, key, value):
        self.key_values[key] = value

    def __getitem__(self, key):
        return self.key_values[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("switch is read only")
