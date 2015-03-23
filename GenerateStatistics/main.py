import os

import constants as const
from utils import reduce_predicates, reduce_sort_timings


# Reads array lengths
array_lens = None
with open(const.FILE_ARRAY_LENS, "r+") as f_array_lens:
    array_lens = [int(l) for l in f_array_lens.read().split(const.SEPARATOR)]


reduce_predicates(const.FOLDER_SORT_CORRECTNESS, "correctness" + const.FILE_EXTENSION)
reduce_predicates(const.FOLDER_SORT_STABILITY, "stability" + const.FILE_EXTENSION)


reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_PARALLEL]
)
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_SEQUENTIAL]
)
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_PARALLEL]
)
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_SEQUENTIAL]
)
