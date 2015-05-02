#ifndef CONSTANTS_RADIX_SORT_H
#define CONSTANTS_RADIX_SORT_H

#include "../../Utils/data_types_common.h"


/*
_KO: Key-only
_KV: Key-value
*/

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PADDING 4


/* ----------------- RADIX SORT LOCAL ---------------- */

// How many threads are used per one thread block for local radix sort kernel. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_LOCAL_SORT_KO 128
#define THREADS_LOCAL_SORT_KV 128
#else
#define THREADS_LOCAL_SORT_KO 128
#define THREADS_LOCAL_SORT_KV 128
#endif
// How many elements are processed by one thread in local radix sort.
// Has to be divisible by 2. Min value is 1, Max value is 8.
#if DATA_TYPE_BITS == 32
#define ELEMS_LOCAL_KO 6
#define ELEMS_LOCAL_KV 4
#else
#define ELEMS_LOCAL_KO 3
#define ELEMS_LOCAL_KV 2
#endif


/* ------------------ GENERATE BUCKET ---------------- */

// How many threads are used per one thread block for kernel, which generates bucket sizes.
// Has to be power of 2. Number of elements processed by one thread is implicitly specified with:
// "(THREADS_LOCAL_SORT * ELEMS_LOCAL) / THREADS_GEN_BUCKETS"
#if DATA_TYPE_BITS == 32
#define THREADS_GEN_BUCKETS_KO 128
#define THREADS_GEN_BUCKETS_KV 128
#else
#define THREADS_GEN_BUCKETS_KO 128
#define THREADS_GEN_BUCKETS_KV 128
#endif


/* ---------------- RADIX SORT GLOBAL ---------------- */

// How many threads are used per one thread block for global radix sort kernel. Has to be power of 2.
// Number of elements processed by one thread is implicitly specified with:
// "(THREADS_LOCAL_SORT * ELEMS_LOCAL) / THREADS_GLOBAL_SORT"
#if DATA_TYPE_BITS == 32
#define THREADS_GLOBAL_SORT_KO 128
#define THREADS_GLOBAL_SORT_KV 256
#else
#define THREADS_GLOBAL_SORT_KO 128
#define THREADS_GLOBAL_SORT_KV 128
#endif


/* ---------- PARALLEL ALGORITHM PARAMETERS ---------- */

// How many bits is the one radix digit made of (one digit is processed in one iteration).
#if DATA_TYPE_BITS == 32
#define BIT_COUNT_PARALLEL_KO 4
#define BIT_COUNT_PARALLEL_KV 4
#else
#define BIT_COUNT_PARALLEL_KO 4
#define BIT_COUNT_PARALLEL_KV 4
#endif


/* --------- SEQUENTIAL ALGORITHM PARAMETERS --------- */

// How many bits is the one radix digit made of (one digit is processed in one iteration).
#if DATA_TYPE_BITS == 32
#define BIT_COUNT_SEQUENTIAL_KO 8
#define BIT_COUNT_SEQUENTIAL_KV 8
#else
#define BIT_COUNT_SEQUENTIAL_KO 8
#define BIT_COUNT_SEQUENTIAL_KV 8
#endif

#endif
