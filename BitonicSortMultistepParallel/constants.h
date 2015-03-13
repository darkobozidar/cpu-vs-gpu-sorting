#ifndef CONSTANTS_BITONIC_SORT_MULTISTEP_PARALLEL_H
#define CONSTANTS_BITONIC_SORT_MULTISTEP_PARALLEL_H

/*
_KO_MSP:  Key-only
_KV_MSP:  Key-value
_MSP: Bitonic sort multistep parallel
*/

/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_BITONIC_SORT_KO_MSP 128
#define THREADS_BITONIC_SORT_KV_MSP 128
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_BITONIC_SORT_KO_MSP 4
#define ELEMS_THREAD_BITONIC_SORT_KV_MSP 4


/* -------------- MULTISTEP MERGE KERNEL ------------- */

// How many threads are used per one thread block in multistep kernel. Has to be power of 2.
#define THREADS_MULTISTEP_MERGE_KO_MSP 512
#define THREADS_MULTISTEP_MERGE_KV_MSP 512
// How much is the biggest allowed multistep - how many elements are sorted by one thread.
// Min value is 1, max value is 6.
#define MAX_MULTI_STEP_KO_MSP 5
#define MAX_MULTI_STEP_KV_MSP 4


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_GLOBAL_MERGE_KO_MSP 256
#define THREADS_GLOBAL_MERGE_KV_MSP 256
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_GLOBAL_MERGE_KO_MSP 4
#define ELEMS_THREAD_GLOBAL_MERGE_KV_MSP 2


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_LOCAL_MERGE_KO_MSP 128
#define THREADS_LOCAL_MERGE_KV_MSP 128
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_LOCAL_MERGE_KO_MSP 4
#define ELEMS_THREAD_LOCAL_MERGE_KV_MSP 4

#endif
