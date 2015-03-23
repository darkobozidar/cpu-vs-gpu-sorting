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
# Folder, where reduction of sort correctness, stability and sort times are saved.
FOLDER_SORT_REDUCTION = "%s%s" % (FOLDER_SORT_ROOT, "ReducedResults/")


# FILES

# File extension.
FILE_EXTENSION = ".txt"
# Separator between elements in file.
SEPARATOR = '\t'
# New line character in file
FILE_NEW_LINE_CHAR = '\n'
# File, wher array lengths are saved
FILE_ARRAY_LENS = "%s%s%s" % (FOLDER_SORT_TEMP, "array_lengths", FILE_EXTENSION)


# GENERAL

# Substring which appeares in file names for sorting key-only
SORT_KEY_ONLY = "key_only"
# Substring which appeares in file names for sorting key-value
SORT_KEY_VALUE = "key_value"
# Substring which appeares in file names for sequential sorts
SORT_SEQUENTIAL = "sequential"
# Substring which appeares in file names for parallel sorts
SORT_PARALLEL = "parallel"
