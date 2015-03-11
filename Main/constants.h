#ifndef MAIN_CONSTANTS_H
#define MAIN_CONSTANTS_H


/* ---------------------- FOLDERS -------------------- */

// Folder where all statistics and temporary files are saved. This is the root folder.
#define FOLDER_SORT_ROOT "../SortStatistics/"
// Temporary folder, where unsorted and sorted arrays are saved into file.
#define FOLDER_SORT_TEMP FOLDER_SORT_ROOT "SortTemp/"
// Folder, where sort execution times are saved.
#define FOLDER_SORT_TIMERS "Time/"
// Folder, where sort correctness statuses are saved.
#define FOLDER_SORT_CORRECTNESS "Correctness/"
// Folder, where sort stability statuses are saved.
#define FOLDER_SORT_STABILITY "Stability/"


/* ----------------------- FILES --------------------- */

// File extension.
#define FILE_EXTENSION ".txt"
// Separator between elements in file.
#define FILE_SEPARATOR_CHAR '\t'
// New line character in file
#define FILE_NEW_LINE '\n'
// File where unsorted array is saved.
#define FILE_UNSORTED_ARRAY FOLDER_SORT_TEMP "unsorted_array" FILE_EXTENSION
// File where correctly sorted array is saved.
#define FILE_SORTED_ARRAY FOLDER_SORT_TEMP "correctly_sorted_array" FILE_EXTENSION

#endif
