#ifndef CONSTANTS_H
#define CONSTANTS_H


/* ----------------- MERGE SORT KERNEL --------------- */

// How many threads are used per one thread block for merge sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_MERGE_SORT 512
// How many elements are processed by one thread in merge sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_MERGE_SORT 2


/* -------------- GENERATE SAMPLES KERNEL ------------ */

// How many threads are used per one thread block for generating samples kernel. Has to be power of 2.
#define THREADS_PER_GEN_SAMPLES 256


/* --------------- GENERATE RANKS KERNEL ------------- */

// How many threads are used per one thread block for generating samples kernel. Has to be power of 2.
#define THREADS_PER_GEN_RANKS 128


#define SHARED_MEM_SIZE 1024
#define SUB_BLOCK_SIZE 256

#endif
