#ifndef CONSTANTS_H
#define CONSTANTS_H


/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 4


/* ----------------- RADIX SORT LOCAL ---------------- */

// How many threads are used per one thread block for local radix sort kernel. Has to be power of 2.
#define THREADS_PER_LOCAL_SORT 128
// How many elements are processed by one thread in local radix sort.
// Has to be divisable by 2. Min value is 1, Max value is 8.
#define ELEMS_PER_THREAD_LOCAL 6


/* ------------------ GENERATE BUCKET ---------------- */

// How many threads are used per one thread block for kernel, which generates bucket sizes.
// Has to be power of 2. Number of elements processed by one thread is implicitly specified with:
// "(THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL) / THREADS_PER_GEN_BUCKETS"
#define THREADS_PER_GEN_BUCKETS 128


/* ---------------- RADIX SORT GLOBAL ---------------- */

// How many threads are used per one thread block for global radix sort kernel. Has to be power of 2.
// Number of elements processed by one thread is implicitly specified with:
// "(THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL) / THREADS_PER_GLOBAL_SORT"

#define THREADS_PER_GLOBAL_SORT 128


/* ---------- PARALLEL ALGORITHM PARAMETERS ---------- */

// How many bits is the one radix diggit made of (one diggit is processed in one iteration).
#define BIT_COUNT_PARALLEL 4
// Radix value - number of all possible different diggit values.
#define RADIX_PARALLEL (1 << BIT_COUNT_PARALLEL)
// Radix mask needed to to perform logical "& (AND)" operation in order to extract diggit from number.
#define RADIX_MASK_PARALLEL ((1 << BIT_COUNT_PARALLEL) - 1)


/* --------- SEQUENTIAL ALGORITHM PARAMETERS --------- */

// How many bits is the one radix diggit made of (one diggit is processed in one iteration).
#define BIT_COUNT_SEQUENTIAL 8
// Radix value - number of all possible different diggit values.
#define RADIX_SEQUENTIAL (1 << BIT_COUNT_SEQUENTIAL)
// Radix mask needed to to perform logical "& (AND)" operation in order to extract diggit from number.
#define RADIX_MASK_SEQUENTIAL ((1 << BIT_COUNT_SEQUENTIAL) - 1)


/* ------------ GENERAL DEVICE PARAMETERS ----------- */

// These constants are needed in order to run C++ "templates", because variables cannot be used
// How many threads are in warp (depending on device - for future compatibility)
#define WARP_SIZE 32

#endif
