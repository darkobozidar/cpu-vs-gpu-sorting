#ifndef CONSTANTS_BITONIC_SORT_H
#define CONSTANTS_BITONIC_SORT_H

/*
_KO: Key-only
_KV: Key-value
*/

/* ---------------- BITONIC SORT KERNEL -------------- */
// KO: key only, KV: key-value

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_BITONIC_SORT_KO 128
#define THREADS_BITONIC_SORT_KV 128
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_BITONIC_SORT_KO 4
#define ELEMS_BITONIC_SORT_KV 4


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_GLOBAL_MERGE_KO 256
#define THREADS_GLOBAL_MERGE_KV 256
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_GLOBAL_MERGE_KO 4
#define ELEMS_GLOBAL_MERGE_KV 2


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_LOCAL_MERGE_KO 256
#define THREADS_LOCAL_MERGE_KV 256
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_LOCAL_MERGE_KO 8
#define ELEMS_LOCAL_MERGE_KV 4

#endif