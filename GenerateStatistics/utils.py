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
            file_output = "%s%s%s" % (sort_name, const.FILE_SEPARATOR_CHAR, predicates_true)
            print(file_output, file=file_summary)

    file_summary.close()
