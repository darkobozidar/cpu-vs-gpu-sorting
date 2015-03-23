import os

import constants as const
from utils import generate_summary_predicates, summarize_sort_timings


generate_summary_predicates(const.FOLDER_SORT_CORRECTNESS)
generate_summary_predicates(const.FOLDER_SORT_STABILITY)


summarize_sort_timings(const.FOLDER_SORT_TIMERS, 2 ** 15, [const.SORT_KEY_ONLY, const.SORT_PARALLEL])
