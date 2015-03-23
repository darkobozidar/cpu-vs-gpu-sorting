#ifndef CONSTANTS_MERGE_SORT_H
#define CONSTANTS_MERGE_SORT_H

/*
_KO: Key-only
_KV: Key-value
*/

/* --------------- ALGORITHM PARAMETERS -------------- */
// Max size of sub-blocks being merged.
// Has to be lower or equal than: THREADS_MERGE_SORT * ELEMS_MERGE_SORT
#define SUB_BLOCK_SIZE_KO 256
#define SUB_BLOCK_SIZE_KV 256


/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PADDING 32


/* ----------------- MERGE SORT KERNEL --------------- */

// How many threads are used per one thread block for merge sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_MERGE_SORT_KO 512
#define THREADS_MERGE_SORT_KV 512
// How many elements are processed by one thread in merge sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_MERGE_SORT_KO 4
#define ELEMS_MERGE_SORT_KV 2


/* --------------- GENERATE RANKS KERNEL ------------- */

// How many threads are used per one thread block for generating ranks kernel. Has to be power of 2.
#define THREADS_GEN_RANKS_KO 128
#define THREADS_GEN_RANKS_KV 128


#endif
