#ifndef CONSTANTS_RADIX_SORT_H
#define CONSTANTS_RADIX_SORT_H

/*
_KO: Key-only
_KV: Key-value
*/

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 4


/* ----------------- RADIX SORT LOCAL ---------------- */

// How many threads are used per one thread block for local radix sort kernel. Has to be power of 2.
#define THREADS_PER_LOCAL_SORT_KO 128
#define THREADS_PER_LOCAL_SORT_KV 128
// How many elements are processed by one thread in local radix sort.
// Has to be divisable by 2. Min value is 1, Max value is 8.
#define ELEMS_PER_THREAD_LOCAL_KO 6
#define ELEMS_PER_THREAD_LOCAL_KV 4


/* ------------------ GENERATE BUCKET ---------------- */

// How many threads are used per one thread block for kernel, which generates bucket sizes.
// Has to be power of 2. Number of elements processed by one thread is implicitly specified with:
// "(THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL) / THREADS_PER_GEN_BUCKETS"
#define THREADS_PER_GEN_BUCKETS 128


/* ---------------- RADIX SORT GLOBAL ---------------- */

// How many threads are used per one thread block for global radix sort kernel. Has to be power of 2.
// Number of elements processed by one thread is implicitly specified with:
// "(THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL) / THREADS_PER_GLOBAL_SORT"

#define THREADS_PER_GLOBAL_SORT_KO 128
#define THREADS_PER_GLOBAL_SORT_KV 256


/* ---------- PARALLEL ALGORITHM PARAMETERS ---------- */

// How many bits is the one radix diggit made of (one diggit is processed in one iteration).
#define BIT_COUNT_PARALLEL_KO 4
#define BIT_COUNT_PARALLEL_KV 4
// Radix value - number of all possible different diggit values.
#define RADIX_PARALLEL (1 << BIT_COUNT_PARALLEL)
// Radix mask needed to to perform logical "& (AND)" operation in order to extract diggit from number.
#define RADIX_MASK_PARALLEL ((1 << BIT_COUNT_PARALLEL) - 1)


/* --------- SEQUENTIAL ALGORITHM PARAMETERS --------- */

// How many bits is the one radix diggit made of (one diggit is processed in one iteration).
#define BIT_COUNT_SEQUENTIAL_KO 8
#define BIT_COUNT_SEQUENTIAL_KV 8

#endif
