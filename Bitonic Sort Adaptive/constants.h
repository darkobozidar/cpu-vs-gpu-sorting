#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 256
// How many elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 2

/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 256
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 2


// Has to be greater or equal than THREADS_PER_SORT
#define THREADS_PER_MERGE 256
#define THREADS_PER_INIT_INTERVALS 128
#define THREADS_PER_GEN_INTERVALS 128

#endif
