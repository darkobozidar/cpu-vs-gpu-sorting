import os
import math
import subprocess

import constants as const


def next_power_2(value):
    """For provided value returns the next power of two."""

    return 2 ** (value - 1).bit_length()


def is_power_2(value):
    """Tests if number is power of 2."""

    return (value != 0) and ((value & (value - 1)) == 0)


def create_folder(folder_path):
    """Checks if folder already exists and creates it in case it doesn't exit."""

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def verify_file_name(file_name, file_name_filters=[], file_extension=""):
    """
    Verifies if file name contains all substrings specified in list and if it has correct
    extension.
    """

    is_ok = not any(f_n not in file_name for f_n in file_name_filters)
    is_ok &= file_name.endswith(file_extension)
    return is_ok


def lengths_to_log(array_lens):
    """
    Converts array lengths so logarithm with base 2 and keeps only those, which are not decimals.
    """

    array_len_logs = (math.log(l, 2) for l in array_lens)
    array_len_str = (str(int(l)) if l == int(l) else "" for l in array_len_logs)
    return const.SEPARATOR.join(array_len_str)


def sort_rate(timings, array_len, factor=1000):
    """Calculates number of elements (millions) that algorithm can sort in one second."""

    avg_time = sum(timings) / len(timings)
    return array_len / factor / avg_time
