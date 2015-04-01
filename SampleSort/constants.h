#ifndef CONSTANTS_SAMPLE_SORT_H
#define CONSTANTS_SAMPLE_SORT_H

#include "../Utils/data_types_common.h"


/*
_KO: Key-only
_KV: Key-value
*/

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PADDING 8


/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_BITONIC_SORT_KO 512
#define THREADS_BITONIC_SORT_KV 256
#else
#define THREADS_BITONIC_SORT_KO 256
#define THREADS_BITONIC_SORT_KV 256
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


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_GLOBAL_MERGE_KO 256
#define THREADS_GLOBAL_MERGE_KV 128
#else
#define THREADS_GLOBAL_MERGE_KO 128
#define THREADS_GLOBAL_MERGE_KV 128
#endif
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#if DATA_TYPE_BITS == 32
#define ELEMS_GLOBAL_MERGE_KO 4
#define ELEMS_GLOBAL_MERGE_KV 4
#else
#define ELEMS_GLOBAL_MERGE_KO 2
#define ELEMS_GLOBAL_MERGE_KV 2
#endif


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define THREADS_LOCAL_MERGE_KO 512
#define THREADS_LOCAL_MERGE_KV 256
#else
#define THREADS_LOCAL_MERGE_KO 512
#define THREADS_LOCAL_MERGE_KV 512
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


/* ----------------- SAMPLE INDEXING ----------------- */

// Has to be greater or equal than NUM_SAMPLES. Has to be multiple of NUM_SAMPLES.
#if DATA_TYPE_BITS == 32
#define THREADS_SAMPLE_INDEXING_KO 128
#define THREADS_SAMPLE_INDEXING_KV 128
#else
#define THREADS_SAMPLE_INDEXING_KO 128
#define THREADS_SAMPLE_INDEXING_KV 128
#endif


/* ---------------- BUCKETS RELOCATION --------------- */

// How many threads are used per one thread block in kernel for buckets relocation. Has to be power of 2.
// Also has to be greater or equal than NUM_SAMPLES. Has to be multiple of NUM_SAMPLES.
#if DATA_TYPE_BITS == 32
#define THREADS_BUCKETS_RELOCATION_KO 128
#define THREADS_BUCKETS_RELOCATION_KV 128
#else
#define THREADS_BUCKETS_RELOCATION_KO 128
#define THREADS_BUCKETS_RELOCATION_KV 128
#endif


/* ---------- PARALLEL ALGORITHM PARAMETERS ---------- */
// Has to be lower or equal than multiplication of THREADS_BITONIC_SORT * ELEMS_BITONIC_SORT.
// Has to be power of 2.
#if DATA_TYPE_BITS == 32
#define NUM_SAMPLES_PARALLEL_KO 32
#define NUM_SAMPLES_PARALLEL_KV 32
#else
#define NUM_SAMPLES_PARALLEL_KO 32
#define NUM_SAMPLES_PARALLEL_KV 32
#endif


/* --------- SEQUENTIAL ALGORITHM PARAMETERS --------- */

// How many splitters are used for buckets. From "N" splitters "N + 1" buckets are created.
#if DATA_TYPE_BITS == 32
#define NUM_SPLITTERS_SEQUENTIAL_KO 16
#define NUM_SPLITTERS_SEQUENTIAL_KV 16
#else
#define NUM_SPLITTERS_SEQUENTIAL_KO 16
#define NUM_SPLITTERS_SEQUENTIAL_KV 16
#endif

// How many extra samples are taken for every splitter. Increases the quality of splitters (samples
// get sorted and only "NUM_SPLITTERS_SEQUENTIAL" splitters are taken from sorted array of samples).
#if DATA_TYPE_BITS == 32
#define OVERSAMPLING_FACTOR_KO 4
#define OVERSAMPLING_FACTOR_KV 4
#else
#define OVERSAMPLING_FACTOR_KO 4
#define OVERSAMPLING_FACTOR_KV 4
#endif

// Threshold, when small sort is applied (in our case merge sort). Has to be greater or equal than
// "NUM_SAMPLES_SEQUENTIAL".
#if DATA_TYPE_BITS == 32
#define SMALL_SORT_THRESHOLD_KO (1 << 15)
#define SMALL_SORT_THRESHOLD_KV (1 << 15)
#else
#define SMALL_SORT_THRESHOLD_KO (1 << 15)
#define SMALL_SORT_THRESHOLD_KV (1 << 14)
#endif

#endif
