#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ---------------- MIN/MAX REDUCTION --------------- */

// How many threads are in each thread block when running min/max reduction. Has to be power of 2.
#define THREADS_PER_REDUCTION 256
// How many elements are processed by each thread in min/max reduction. Has to be power of 2.
#define ELEMENTS_PER_THREAD_REDUCTION 32
// Threashold when reduction on device stops and result is coppied to host. Reduction is finnished on host.
#define THRESHOLD_REDUCTION (1 << 12)

/* ---------------- GLOBAL QUICKSORT ---------------- */

// How many threads are in each thread block when running global quicksort kernel. Has to be power of 2.
#define THREADS_PER_SORT_GLOBAL 256
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#define ELEMENTS_PER_THREAD_GLOBAL 4
// Threshold size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#define THRESHOLD_PARTITION_SIZE_GLOBAL (1 << 11)


/* ----------------- LOCAL QUICKSORT ---------------- */

// How many threads are in each thread block when running local quicksort kernel. Has to be power of 2.
// It is reasonable that is is lower than "THREADS_PER_SORT_GLOBAL * ELEMENTS_PER_THREAD_GLOBAL".
#define THREADS_PER_SORT_LOCAL 256
// Threshold for sequence size in local quick sort, when bitonic sort is used.
// TODO rename to threashold
#define BITONIC_SORT_SIZE_LOCAL 512

// TODO min-max val depending on data type

#endif
