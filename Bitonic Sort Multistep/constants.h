#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ---------------- BITONIC SORT KERNEL -------------- */
// Bottom 2 constants are limited by shared memory size

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 512
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 4


#define THREADS_PER_MERGE 256
#define MAX_THREADS_PER_MULTISTEP 128
#define MAX_MULTI_STEP 4

#endif
