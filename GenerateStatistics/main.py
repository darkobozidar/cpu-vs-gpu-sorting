import os

import constants as const
from statistics import test_sorts, reduce_predicates, reduce_sort_timings


# Tests sorts on specified interval
test_sorts(const.FILE_SORT_EXE, (1 << 15), (1 << 25), 30)
print("\n\n")

# Reads array lengths
array_lens = None
with open(const.FILE_ARRAY_LENS, "r+") as f_array_lens:
    array_lens = [int(l) for l in f_array_lens.read().split(const.FILE_NEW_LINE_CHAR)[:-1]]

# Reduces sort correctness and stability
print("Reducing sort correctness")
reduce_predicates(const.FOLDER_SORT_CORRECTNESS, "correctness" + const.FILE_EXTENSION)
print("Reducing sort stability")
reduce_predicates(const.FOLDER_SORT_STABILITY, "stability" + const.FILE_EXTENSION)

# Reduces timings
print("Reducing sort timings for key only parallel sort")
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_PARALLEL]
)

print("Reducing sort timings for key only sequential sort")
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_ONLY, const.SORT_SEQUENTIAL]
)

print("Reducing sort timings for key value parallel sort")
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_PARALLEL]
)

print("Reducing sort timings for key value sequential sort")
reduce_sort_timings(
    const.FOLDER_SORT_TIMERS, array_lens, [const.SORT_KEY_VALUE, const.SORT_SEQUENTIAL]
)
