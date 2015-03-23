# FOLDERS

# Folder where all statistics and temporary files are saved. This is the root folder.
FOLDER_SORT_ROOT = "../SortStatistics/"
# Temporary folder, where unsorted and sorted arrays are saved into file.
FOLDER_SORT_TEMP = "%s%s" % (FOLDER_SORT_ROOT, "SortTemp/")
# Folder, where sort execution times are saved.
FOLDER_SORT_TIMERS = "%s%s" % (FOLDER_SORT_ROOT, "Time/")
# Folder, where sort correctness statuses are saved.
FOLDER_SORT_CORRECTNESS = "%s%s" % (FOLDER_SORT_ROOT, "Correctness/")
# Folder, where sort stability statuses are saved.
FOLDER_SORT_STABILITY = "%s%s" % (FOLDER_SORT_ROOT, "Stability/")


# FILES

# File extension.
FILE_EXTENSION = ".txt"
# Separator between elements in file.
SEPARATOR = '\t'
# New line character in file
FILE_NEW_LINE_CHAR = '\n'
# File, wher array lengths are saved
FILE_ARRAY_LENS = "%s%s%s" % (FOLDER_SORT_TEMP, "array_lengths", FILE_EXTENSION)
# File name, where summary of predicates for sort correctness and stability are saved
FILE_SUMMARY = "%s%s" % ("Summary", FILE_EXTENSION)

SORT_KEY_ONLY = "key_only"
SORT_KEY_VALUE = "key_value"
SORT_SEQUENTIAL = "sequential"
SORT_PARALLEL = "parallel"
