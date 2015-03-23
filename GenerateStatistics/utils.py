import os

import constants as const


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


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
            sort_output = "%s%s%s" % (sort_name, const.SEPARATOR, predicates_true)
            print(sort_output, file=file_summary)

    file_summary.close()


def calc_avg_sort_rate(timings, array_len):
    avg_time = sum(timings) / len(timings)
    return array_len / 1000 / avg_time


def summarize_sort_timings(folder_name, array_lens, file_name_filters=[]):
    for distribution in os.listdir(folder_name):
        folder_dist = "%s%s/" % (folder_name, distribution)
        folder_summary = "%s%s" %(folder_dist, "Summary/")
        create_folder(folder_summary)

        file_name_output = "%s%s%s" % (folder_summary, '_'.join(file_name_filters), const.FILE_EXTENSION)
        file_output = open(file_name_output, "w+")

        header = "%s%s" % (const.SEPARATOR, const.SEPARATOR.join(str(l) for l in array_lens))
        print(header, file=file_output)

        for file_name_sort in os.listdir(folder_dist):
            # Filters file name with provided filters
            if any(f not in file_name_sort for f in file_name_filters):
                continue
            if not file_name_sort.endswith(const.FILE_EXTENSION):
                continue

            with open("%s%s" % (folder_dist, file_name_sort), "r") as file_sort:
                content = file_sort.read()
                lines = content.split(const.FILE_NEW_LINE_CHAR)[:-1]
                timings = [[float(t) for t in l.split(const.SEPARATOR)] for l in lines]
                timings_summary = [calc_avg_sort_rate(t, l) for t, l in zip(timings, array_lens)]

            sort_name = str(file_name_sort)
            for file_filter in file_name_filters:
                sort_name = sort_name.replace(file_filter, "")

            sort_name = sort_name[:-len(const.FILE_EXTENSION)]
            sort_name = " ".join(s for s in sort_name.split("_") if s)
            timings_output = const.SEPARATOR.join(str(t).replace(".", ",") for t in timings_summary)
            output = "%s%s%s" % (sort_name, const.SEPARATOR, timings_output)

            print(output, file=file_output)

        file_output.close()
