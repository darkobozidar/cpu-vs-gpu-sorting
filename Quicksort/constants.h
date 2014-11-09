#ifndef CONSTANTS_H
#define CONSTANTS_H


/* --------------- ALGORITHM PARAMETERS -------------- */
// For GLOBAL quicksrot it designates whether:
// - VAL 0: PIVOT value is used as MAX value of newly generated LOWER sequence and MIN value of newly
//          generated GREATER sequence (FASTER, but possibility of WRONG min/max value)
// - VAL 1: min/max reduction is performed in order to find MAX value of newly generated LOWER sequence
//          and MIN value of newly generated GREATER sequence (SLOWER, but ALWAYS correct min/max value)
#define USE_REDUCTION_IN_GLOBAL_SORT 1


/* ---------------- MIN/MAX REDUCTION --------------- */

// How many threads are in each thread block when running min/max reduction. Has to be power of 2.
#define THREADS_PER_REDUCTION 128
// How many elements are processed by each thread in min/max reduction. Has to be power of 2.
#define ELEMENTS_PER_THREAD_REDUCTION 64
// Threshold of array length, when reduction is performed on DEVICE instead of HOST.
#define THRESHOLD_REDUCTION (1 << 13)


/* ---------------- GLOBAL QUICKSORT ---------------- */

// How many threads are in each thread block when running global quicksort kernel. Has to be power of 2.
#define THREADS_PER_SORT_GLOBAL 128
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#define ELEMENTS_PER_THREAD_GLOBAL 5
// Threshold size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#define THRESHOLD_PARTITION_SIZE_GLOBAL (1 << 11)


/* ----------------- LOCAL QUICKSORT ---------------- */

// How many threads are in each thread block when running local quicksort kernel. Has to be power of 2.
// It is reasonable that is is lower than "THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL".
#define THREADS_PER_SORT_LOCAL 256
// Threshold for sequence size in local quick sort, when bitonic sort is used.
#define THRESHOLD_BITONIC_SORT_LOCAL 512


/* ------------ GENERAL DEVICE PARAMETERS ----------- */

// These constants are needed in order to run C++ "templates", because variables cannot be used
// How many threads are in warp (depending on device - for future compatibility)
#define WARP_SIZE 32
// Log¡2 of WARP_SIZE for faster computation because of left/right bit-shifts
#define WARP_SIZE_LOG 5

#endif
