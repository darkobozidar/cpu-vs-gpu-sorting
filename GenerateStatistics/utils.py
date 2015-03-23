import os

import constants as const


def generate_summary_predicates(folder_name, file_name_filter=""):
    """
    Goes through all files in folder, reads content and checks, if all predicates in file are true.
    Writes summary to file in same folder.
    """

    file_name_summary = "%s%s" % (folder_name, const.FILE_SUMMARY)
    file_summary = open(file_name_summary, "w+")

    for file_name in os.listdir(folder_name):
        if file_name_filter not in file_name:
            continue

        with open(folder_name + file_name) as file_sort:
            file_content = file_sort.read()
            predicates_true = all(int(predicate) for predicate in file_content.split())

            sort_name = file_name.replace("_", " ")
            sort_output = "%s%s%s" % (sort_name, const.FILE_SEPARATOR_CHAR, predicates_true)
            print(sort_output, file=file_summary)

    file_summary.close()

def get_sort_timings(folder_dist, file_name_filters=[]):
    sorts = []

    for file_name_sort in os.listdir(folder_dist):
        # Filters file name with provided filters
        if any(f not in file_name_sort for f in file_name_filters):
            continue
        if not file_name_sort.endswith(const.FILE_EXTENSION):
            continue

        with open("%s%s" % (folder_dist, file_name_sort), "r") as file_sort:
            content = file_sort.read()
            lines = content.split(const.FILE_NEW_LINE_CHAR)[:-1]
            timings = [[float(t) for t in l.split(const.FILE_SEPARATOR_CHAR)] for l in lines]
            sorts.append((file_name_sort.replace("_", " "), timings))

    sorts.sort(key=lambda el: el[0])
    return sorts


def calc_avg_sort_rate(timings, array_len):
    avg_time = sum(timings) / len(timings)
    return array_len / 1000 / avg_time


def summarize_sort_timings(folder_name, start_array_len, file_name_filters=[]):
    for distribution in os.listdir(folder_name):
        folder_dist = "%s%s/" % (folder_name, distribution)
        sorts = get_sort_timings(folder_dist, file_name_filters)

        folder_summary = "%s%s" %(folder_dist, "Summary/")
        if not os.path.exists(folder_summary):
            os.makedirs(folder_summary)

        file_name_output = "%s%s%s" % (folder_summary, '_'.join(file_name_filters), const.FILE_EXTENSION)
        file_output = open(file_name_output, "w+")
        num_timing_lines, array_len = len(sorts[0][1]), start_array_len

        sorts_header = const.FILE_SEPARATOR_CHAR.join(name for name, _ in sorts)
        header = "%s%s%s" % ("Length", const.FILE_SEPARATOR_CHAR, sorts_header)
        print(header, file=file_output)

        for line in range(num_timing_lines):
            line_summary = [str(line)]

            for _, timings in sorts:
                output = str(calc_avg_sort_rate(timings[line], array_len))
                line_summary.append(output.replace(".", ","))

            array_len *= 2
            print(const.FILE_SEPARATOR_CHAR.join(line_summary), file=file_output)

        file_output.close()
