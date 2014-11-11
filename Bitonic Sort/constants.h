#ifndef CONSTANTS_H
#define CONSTANTS_H


/* ---------------- BITONIC SORT KERNEL -------------- */
// Bottom 2 constants are limited by shared memory size

// How many threads are used per one thread block for (local) bitonic sort, which is performed entirely
// in shared memory.
#define THREADS_PER_BITONIC_SORT 2
// How many elements are processed by one thread in (local) bitonic sort kernel. Min value is 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 2


#define THREADS_PER_LOCAL_MERGE 512
#define THREADS_PER_GLOBAL_MERGE 128

#endif
