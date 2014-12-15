#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 4


/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 128
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 4

#define THREADS_PER_INIT_INTERVALS 128
#define THREADS_PER_GEN_INTERVALS 128


/* ------------------- BITONIC MERGE ----------------- */
// "THREADS_PER_MERGE * ELEMS_PER_MERGE" has to be lower or equal than
// "THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT"

// How many threads are used per on thread block for bitonic merge. Has to be power of 2.
#define THREADS_PER_MERGE 128
// How many elements are processed by one thread in bitonic merge kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_MERGE 4

#endif
