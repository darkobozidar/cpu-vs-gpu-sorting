#ifndef CONSTANTS_BITONIC_SORT_MULTISTEP_H
#define CONSTANTS_BITONIC_SORT_MULTISTEP_H

#include "../Utils/data_types_common.h"


/*
_KO:  Key-only
_KV:  Key-value
*/

/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_BITONIC_SORT_KO 128
#define THREADS_BITONIC_SORT_KV 128
#else
#define THREADS_BITONIC_SORT_KO 128
#define THREADS_BITONIC_SORT_KV 128
#endif
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisible by 2.
#if DATA_TYPE_BITS == 32
#define ELEMS_BITONIC_SORT_KO 4
#define ELEMS_BITONIC_SORT_KV 4
#else
#define ELEMS_BITONIC_SORT_KO 4
#define ELEMS_BITONIC_SORT_KV 2
#endif


/* -------------- MULTISTEP MERGE KERNEL ------------- */

// How many threads are used per one thread block in multistep kernel. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_MULTISTEP_MERGE_KO 512
#define THREADS_MULTISTEP_MERGE_KV 512
#else
#define THREADS_MULTISTEP_MERGE_KO 256
#define THREADS_MULTISTEP_MERGE_KV 256
#endif
// How much is the biggest allowed multistep - how many elements are sorted by one thread.
#if DATA_TYPE_BITS == 32
// Min value is 1, max value is 6.
#define MAX_MULTI_STEP_KO 5
// Min value is 1, max value is 5.
#define MAX_MULTI_STEP_KV 4
#else
// Min value is 1, max value is 6.
#define MAX_MULTI_STEP_KO 3
// Min value is 1, max value is 5.
#define MAX_MULTI_STEP_KV 3
#endif


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_GLOBAL_MERGE_KO 256
#define THREADS_GLOBAL_MERGE_KV 256
#else
#define THREADS_GLOBAL_MERGE_KO 128
#define THREADS_GLOBAL_MERGE_KV 128
#endif
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisible by 2.
#if DATA_TYPE_BITS == 32
#define ELEMS_GLOBAL_MERGE_KO 4
#define ELEMS_GLOBAL_MERGE_KV 2
#else
#define ELEMS_GLOBAL_MERGE_KO 2
#define ELEMS_GLOBAL_MERGE_KV 2
#endif


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_LOCAL_MERGE_KO 128
#define THREADS_LOCAL_MERGE_KV 128
#else
#define THREADS_LOCAL_MERGE_KO 256
#define THREADS_LOCAL_MERGE_KV 256
#endif
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisible by 2.
#if DATA_TYPE_BITS == 32
#define ELEMS_LOCAL_MERGE_KO 4
#define ELEMS_LOCAL_MERGE_KV 4
#else
#define ELEMS_LOCAL_MERGE_KO 4
#define ELEMS_LOCAL_MERGE_KV 2
#endif

#endif
