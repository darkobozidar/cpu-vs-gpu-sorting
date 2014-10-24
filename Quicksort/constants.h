#ifndef CONSTANTS_H
#define CONSTANTS_H

// TODO remove (calculate it from MIN_PARTITION_SIZE)
#define MAX_SEQUENCES 1024  // Maximum number of sequences, which get produced by global quicksort

// Minimum size until sequence can still get partitioned. When sequence's length is lower or equal to this
// constant, than it stops to be partitioned by global quicksort. Has to be power of 2.
#define MIN_PARTITION_SIZE_GLOBAL 8
// How many threads are in each thread block in global quicksort. Has to be power of 2.
#define THREADS_PER_SORT_GLOBAL 2
// How many elements are processed by each thread in global quicksort. Has to be power of 2.
#define ELEMENTS_PER_THREAD_GLOBAL 2

#define THREADS_PER_SORT_LOCAL 2  // Thread block size for local quicksort. Has to be power of 2.
#define BITONIC_SORT_SIZE_LOCAL 8 // Threshold of sub-block size, when bitonic sort is used

#endif
