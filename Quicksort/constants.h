#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ---------------- MIN/MAX REDUCTION --------------- */

// How many threads are in each thread block when running min/max reduction. Has to be power of 2.
#define THREADS_PER_REDUCTION 2
// How many elements are processed by each thread in min/max reduction. Has to be power of 2.
#define ELEMENTS_PER_THREAD_REDUCTION 2
// Threashold when reduction on device stops and result is coppied to host. Reduction is finnished on host.
#define THRESHOLD_REDUCTION 2

/* ---------------- GLOBAL QUICKSORT ---------------- */

// How many threads are in each thread block when running global quicksort kernel. Has to be power of 2.
#define THREADS_PER_SORT_GLOBAL 16
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#define ELEMENTS_PER_THREAD_GLOBAL 2
// Minimum size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#define MIN_PARTITION_SIZE_GLOBAL 2


/* ----------------- LOCAL QUICKSORT ---------------- */

// How many threads are in each thread block when running local quicksort kernel. Has to be power of 2
// and lower or equal than THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL.
#define THREADS_PER_SORT_LOCAL 2
// Threshold for sequence size in local quick sort, when bitonic sort is used.
#define BITONIC_SORT_SIZE_LOCAL 4

// TODO min-max val depending on data type

#endif
