#ifndef CONSTANTS_H
#define CONSTANTS_H


/* ---------------- BITONIC SORT KERNEL -------------- */
// Bottom 2 constants are limited by shared memory size

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


#endif
