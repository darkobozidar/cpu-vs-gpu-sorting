import math
import subprocess
import os

import constants as const
from utils import (
    next_power_2, is_power_2, create_folder, verify_file_name, lengths_to_log, sort_rate
)


def array_len_gen(array_len_start, array_len_end, interval_split=2):
    """
    Generator for array lengths. Interval_split specifies, how many times should interval be
    sampled/tested between array length 2^n and 2^(n + 1).
    """

    array_len = array_len_start

    while array_len < array_len_end:
        step = int(array_len / interval_split)

        for l in range(array_len, next_power_2(array_len + 1) - step + 1, step):
            yield l

        array_len = next_power_2(array_len + 1)

    yield array_len


def test_sorts(exe_path, array_len_start, array_len_end, test_repetitions,
               sort_order=const.ORDER_ASC):
    """Tests sorting algorithms for provided array lengths."""

    for array_len in array_len_gen(array_len_start, array_len_end):
        subprocess.call([exe_path, str(array_len), str(test_repetitions), str(sort_order)])


def reduce_predicates(folder_name_pred, output_file_name, file_name_filters=""):
    """
    Goes through all files in folder, reads content and checks, if all predicates in file are true.
    Writes reduced result to output file.
    """

    # Creates output file
    create_folder(const.FOLDER_SORT_REDUCTION)
    file_name_reduction = "%s%s" % (const.FOLDER_SORT_REDUCTION, output_file_name)
    file_reduction = open(file_name_reduction, "w+")

    for file_name in os.listdir(folder_name_pred):
        if not verify_file_name(file_name, file_name_filters, const.FILE_EXTENSION):
            continue

        # Reduces all predicates to only one predicate "True" of "False"
        with open(folder_name_pred + file_name) as file_sort:
            file_content = file_sort.read()
            predicates_true = all(int(predicate) for predicate in file_content.split())

            sort_name = file_name.replace("_", " ")
            sort_output = "%s%s%s" % (sort_name, const.SEPARATOR, predicates_true)
            print(sort_output, file=file_reduction)

    file_reduction.close()


def reduce_sort_timings(folder_name, array_lens, file_name_filters=[], reduce_func=sort_rate):
    """Reduces sort timings and outputs them to files."""

    for distribution in os.listdir(folder_name):
        folder_dist_input = "%s%s/" % (folder_name, distribution)
        folder_dist_output = "%s%s/" % (const.FOLDER_SORT_REDUCTION, distribution)
        create_folder(folder_dist_output)

        # Creates output file
        file_name_output = "%s%s%s" % (
            folder_dist_output, '_'.join(file_name_filters), const.FILE_EXTENSION
        )
        file_output = open(file_name_output, "w+")

        # Saves header to output file
        header = "%s%s" % (const.SEPARATOR, lengths_to_log(array_lens))
        print(header, file=file_output)

        for file_name_sort in os.listdir(folder_dist_input):
            if not verify_file_name(file_name_sort, file_name_filters, const.FILE_EXTENSION):
                continue

            # Reduces sort timings
            with open("%s%s" % (folder_dist_input, file_name_sort), "r") as file_sort:
                content = file_sort.read()
                lines = content.split(const.FILE_NEW_LINE_CHAR)[:-1]
                timings = [[float(t) for t in l.split(const.SEPARATOR)] for l in lines]
                timings_reduced = [reduce_func(t, l) for t, l in zip(timings, array_lens)]

            # Generates sort name
            sort_name = str(file_name_sort)
            for file_filter in file_name_filters:
                sort_name = sort_name.replace(file_filter, "")

            # Outputs sort timings
            sort_name = sort_name[:-len(const.FILE_EXTENSION)]
            sort_name = " ".join(s for s in sort_name.split("_") if s)
            timings_output = const.SEPARATOR.join(str(t).replace(".", ",") for t in timings_reduced)
            output = "%s%s%s" % (sort_name, const.SEPARATOR, timings_output)

            print(output, file=file_output)

        file_output.close()
