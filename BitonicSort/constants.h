#ifndef CONSTANTS_BITONIC_SORT_PARALLEL_H
#define CONSTANTS_BITONIC_SORT_PARALLEL_H

/*
_KO_BSP: Key-only
_KV_BSP: Key-value
_BSP: Bitonic sort parallel
*/

/* ---------------- BITONIC SORT KERNEL -------------- */
// KO: key only, KV: key-value

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_BITONIC_SORT_KO_BSP 512
#define THREADS_BITONIC_SORT_KV_BSP 256
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_BITONIC_SORT_KO_BSP 4
#define ELEMS_THREAD_BITONIC_SORT_KV_BSP 4


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_GLOBAL_MERGE_KO_BSP 256
#define THREADS_GLOBAL_MERGE_KV_BSP 256
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_GLOBAL_MERGE_KO_BSP 4
#define ELEMS_THREAD_GLOBAL_MERGE_KV_BSP 2


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_LOCAL_MERGE_KO_BSP 256
#define THREADS_LOCAL_MERGE_KV_BSP 256
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_THREAD_LOCAL_MERGE_KO_BSP 8
#define ELEMS_THREAD_LOCAL_MERGE_KV_BSP 4

#endif
