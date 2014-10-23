#ifndef CONSTANTS_H
#define CONSTANTS_H

#define MAX_SEQUENCES 2  // Maximum number of sequences, which get produced by global quicksort

#define THREADS_PER_SORT_LOCAL 2  // Thread block size for local quicksort. Has to be power of 2.
#define BITONIC_SORT_SIZE_LOCAL 8 // Threshold of sub-block size, when bitonic sort is used

#endif
