#ifndef MAIN_CONSTANTS_H
#define MAIN_CONSTANTS_H


/* ---------------------- FOLDERS -------------------- */

// Folder where all statistics and temporary files are saved. This is the root folder.
#define FOLDER_SORT_STATS "../SortStatistics"
// Temporary folder, where unsorted arrays are saved into file.
#define FOLDER_SORT_TEMP FOLDER_SORT_STATS "/SortTemp"


/* ----------------------- FILES --------------------- */

// File where unsorted array is saved.
#define FILE_UNSORTED_ARRAY FOLDER_SORT_TEMP "/unsorted_array.txt"

#endif
