import os

import constants as const
from utils import generate_summary_predicates, summarize_sort_timings


array_lens = None
with open(const.FILE_ARRAY_LENS, "r+") as f_array_lens:
    array_lens = [int(l) for l in f_array_lens.read().split(const.SEPARATOR)]


generate_summary_predicates(const.FOLDER_SORT_CORRECTNESS)
generate_summary_predicates(const.FOLDER_SORT_STABILITY)


summarize_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_PARALLEL]
)
summarize_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_SEQUENTIAL]
)
summarize_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_PARALLEL]
)
summarize_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_SEQUENTIAL]
)
