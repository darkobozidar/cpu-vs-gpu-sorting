#ifndef CONSTANTS_H
#define CONSTANTS_H


/* --------------- ALGORITHM PARAMETERS -------------- */
// Has to be lower or equal than multiplication of THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT.
// Has to be power of 2.
#define NUM_SAMPLES 2


/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 2
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 2


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_PER_GLOBAL_MERGE 2
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_GLOBAL_MERGE 2


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_PER_LOCAL_MERGE 2
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_LOCAL_MERGE 2


/* ----------------- SAMPLE INDEXING ----------------- */

// Has to be greater or equal than NUM_SAMPLES.
#define THREADS_PER_SAMPLE_INDEXING 4


/* ---------------- BUCKETS RELOCATION --------------- */

// How many threads are used per one thread block in kernel for buckets relocation. Has to be power of 2.
// Also has to be greater or equal than NUM_SAMPLES.
#define THREADS_PER_BUCKETS_RELOCATION (12 * NUM_SAMPLES)

#endif
