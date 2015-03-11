#ifndef MAIN_CONSTANTS_H
#define MAIN_CONSTANTS_H


/* ---------------------- FOLDERS -------------------- */

// Folder where all statistics and temporary files are saved. This is the root folder.
#define FOLDER_SORT_ROOT "../SortStatistics/"
// Temporary folder, where unsorted and sorted arrays are saved into file.
#define FOLDER_SORT_TEMP FOLDER_SORT_ROOT "SortTemp/"
// Folder, where sort execution times are saved.
#define FOLDER_SORT_TIMERS FOLDER_SORT_ROOT "Time/"
// Folder, where sort correctness statuses are saved.
#define FOLDER_SORT_CORRECTNESS FOLDER_SORT_ROOT "Correctness/"
// Folder, where sort stability statuses are saved.
#define FOLDER_SORT_STABILITY FOLDER_SORT_ROOT "Stability/"


/* ----------------------- FILES --------------------- */

// File extension.
#define FILE_EXTENSION ".txt"
// Prefix for log file
#define FILE_LOG_PREFIX "log_"
// Separator between elements in file.
#define FILE_SEPARATOR_CHAR '\t'
// New line character in file
#define FILE_NEW_LINE_CHAR '\n'
// File where unsorted array is saved.
#define FILE_UNSORTED_ARRAY FOLDER_SORT_TEMP "array_unsorted" FILE_EXTENSION
// File where correctly sorted array is saved.
#define FILE_SORTED_ARRAY FOLDER_SORT_TEMP "array_correctly_sorted" FILE_EXTENSION

#endif
