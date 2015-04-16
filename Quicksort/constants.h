#ifndef CONSTANTS_QUICKSORT_H
#define CONSTANTS_QUICKSORT_H

#include "../Utils/data_types_common.h"


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

// Threshold of array length, when reduction is performed on DEVICE instead of HOST.
#if DATA_TYPE_BITS == 32
#define THRESHOLD_PARALLEL_REDUCTION (1 << 13)
#else
#define THRESHOLD_PARALLEL_REDUCTION (1 << 13)
#endif
// How many threads are in each thread block when running min/max reduction. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_REDUCTION 128
#else
#define THREADS_REDUCTION 128
#endif
// How many elements are processed by each thread in min/max reduction. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define ELEMENTS_REDUCTION 64
#else
#define ELEMENTS_REDUCTION 64
#endif


/* ---------------- GLOBAL QUICKSORT ---------------- */

// Threshold size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KO (1 << 11)
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KV (1 << 10)
#else
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KO (1 << 10)
#define THRESHOLD_PARTITION_SIZE_GLOBAL_KV (1 << 10)
#endif
// How many threads are in each thread block when running global quicksort kernel. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_SORT_GLOBAL_KO 128
#define THREADS_SORT_GLOBAL_KV 128
#else
#define THREADS_SORT_GLOBAL_KO 128
#define THREADS_SORT_GLOBAL_KV 128
#endif
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define ELEMENTS_GLOBAL_KO 6  // 8 if USE_REDUCTION_IN_GLOBAL_SORT is 1
#define ELEMENTS_GLOBAL_KV 4  // 8 if USE_REDUCTION_IN_GLOBAL_SORT is 1
#else
#define ELEMENTS_GLOBAL_KO 4
#define ELEMENTS_GLOBAL_KV 2
#endif


/* ----------------- LOCAL QUICKSORT ---------------- */

// Threshold for sequence size in local quick sort, when bitonic sort is used.
#if DATA_TYPE_BITS == 32
#define THRESHOLD_BITONIC_SORT_KO 512
#define THRESHOLD_BITONIC_SORT_KV 256
#else
#define THRESHOLD_BITONIC_SORT_KO 256
#define THRESHOLD_BITONIC_SORT_KV 256
#endif
// How many threads are in each thread block when running local quicksort kernel. Has to be power of 2.
// It is reasonable that is is lower than "THREADS_SORT_GLOBAL * ELEMENTS_GLOBAL".
#if DATA_TYPE_BITS == 32
#define THREADS_SORT_LOCAL_KO 128
#define THREADS_SORT_LOCAL_KV 128
#else
#define THREADS_SORT_LOCAL_KO 128
#define THREADS_SORT_LOCAL_KV 128
#endif

#endif
