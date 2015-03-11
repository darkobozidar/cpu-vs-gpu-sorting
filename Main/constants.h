#ifndef MAIN_CONSTANTS_H
#define MAIN_CONSTANTS_H


/* ---------------------- FOLDERS -------------------- */

// Folder where all statistics and temporary files are saved. This is the root folder.
#define FOLDER_SORT_ROOT "../SortStatistics/"
// Temporary folder, where unsorted and sorted arrays are saved into file.
#define FOLDER_SORT_TEMP FOLDER_SORT_ROOT "SortTemp/"
// Folder, where sort execution times are saved.
#define FOLDER_SORT_TIMERS FOLDER_SORT_ROOT "Timers/"


/* ----------------------- FILES --------------------- */

// File where unsorted array is saved.
#define FILE_UNSORTED_ARRAY FOLDER_SORT_TEMP "unsorted_array.txt"

#endif
