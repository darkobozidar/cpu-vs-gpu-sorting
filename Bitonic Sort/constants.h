#ifndef CONSTANTS_H
#define CONSTANTS_H


/* ---------------- BITONIC SORT KERNEL -------------- */
// Bottom 2 constants are limited by shared memory size
// TODO write, that multiple elements per kernel is not a good idea in bitonic sort. Remove this from code.

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 512
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 2


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_PER_GLOBAL_MERGE 128
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_GLOBAL_MERGE 2


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_PER_LOCAL_MERGE 512
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_LOCAL_MERGE 2


/* ---------------- TESTING CONSTANTS ---------------- */
// How may timers are stopwatched (PARALLEL, SEQUENTIAL and CORRECT). DON'T CHANGE!
#define NUM_STOPWATCHES 3

#endif
