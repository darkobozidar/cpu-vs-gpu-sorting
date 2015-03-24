import os
import math

import constants as const


def create_folder(folder_path):
    """Checks if folder already exists and creates it in case it doesn't exit."""

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def verify_file_name(file_name, file_name_filters=[], file_extension=""):
    """
    Verifies if file name contains all substring specified in list and if it has correct
    exttension.
    """

    is_ok = not any(f_n not in file_name for f_n in file_name_filters)
    is_ok &= file_name.endswith(file_extension)
    return is_ok


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



def sort_rate(timings, array_len, factor=1000):
    """Calculates number of millions of elements that algorithm can sort in once second."""

    avg_time = sum(timings) / len(timings)
    return array_len / factor / avg_time


def lengths_to_log(array_lens):
    """
    Converts array lengths so logarithm with base 2 and keeps only those, which are not decimals.
    """

    array_len_logs = (math.log(l, 2) for l in array_lens)
    array_len_str = (str(int(l)) if l == int(l) else "" for l in array_len_logs)
    return const.SEPARATOR.join(array_len_str)


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


# TODO remove
def test_sorts(folder_name):
    main_files = [f for f in os.listdir(folder_name) if "Main" in f and f.endswith(".exe")]
    main_files_numbers = [int(f.split("_")[1][:-4]) for f in main_files]
    main_files_numbers.sort()

    for m in main_files_numbers:
        os.system('%sMain_%d' % (folder_name, m))
