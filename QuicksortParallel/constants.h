#ifndef CONSTANTS_H
#define CONSTANTS_H


/*
_KO: Key-only
_KV: Key-value
*/

/* --------------- ALGORITHM PARAMETERS -------------- */
// For GLOBAL quicksrot it designates whether:
// - VAL 0: PIVOT value is used as MAX value of newly generated LOWER sequence and MIN value of newly
//          generated GREATER sequence (FASTER, but possibility of WRONG min/max value)
// - VAL 1: min/max reduction is performed in order to find MAX value of newly generated LOWER sequence
//          and MIN value of newly generated GREATER sequence (SLOWER, but ALWAYS correct min/max value)
#define USE_REDUCTION_IN_GLOBAL_SORT 0


/* ---------------- MIN/MAX REDUCTION --------------- */

// How many threads are in each thread block when running min/max reduction. Has to be power of 2.
#define THREADS_PER_REDUCTION_KO 128
#define THREADS_PER_REDUCTION_KV 128
// How many elements are processed by each thread in min/max reduction. Has to be power of 2.
#define ELEMENTS_PER_THREAD_REDUCTION_KO 64
#define ELEMENTS_PER_THREAD_REDUCTION_KV 64
// Threshold of array length, when reduction is performed on DEVICE instead of HOST.
#define THRESHOLD_REDUCTION_KO (1 << 13)
#define THRESHOLD_REDUCTION_KV (1 << 13)


/* ---------------- GLOBAL QUICKSORT ---------------- */

// How many threads are in each thread block when running global quicksort kernel. Has to be power of 2.
#define THREADS_PER_SORT_GLOBAL_KO 128
#define THREADS_PER_SORT_GLOBAL_KV 128
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#define ELEMENTS_PER_THREAD_GLOBAL_KO 6  // 8 if USE_REDUCTION_IN_GLOBAL_SORT is 1
#define ELEMENTS_PER_THREAD_GLOBAL_KV 4  // 8 if USE_REDUCTION_IN_GLOBAL_SORT is 1
// Threshold size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KO (1 << 11)
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KV (1 << 11)


/* ----------------- LOCAL QUICKSORT ---------------- */

// How many threads are in each thread block when running local quicksort kernel. Has to be power of 2.
// It is reasonable that is is lower than "THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL".
#define THREADS_PER_SORT_LOCAL_KO 128
#define THREADS_PER_SORT_LOCAL_KV 256
// Threshold for sequence size in local quick sort, when bitonic sort is used.
#define THRESHOLD_BITONIC_SORT_LOCAL_KO 512
#define THRESHOLD_BITONIC_SORT_LOCAL_KV 512

#endif
