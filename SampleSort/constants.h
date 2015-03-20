#ifndef CONSTANTS_SAMPLE_SORT_H
#define CONSTANTS_SAMPLE_SORT_H

/*
_KO: Key-only
_KV: Key-value
*/

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 8


/* --------------- ALGORITHM PARAMETERS -------------- */
// Has to be lower or equal than multiplication of THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT.
// Has to be power of 2.
#define NUM_SAMPLES_PARALLEL 32


/* ---------------- BITONIC SORT KERNEL -------------- */

// How many threads are used per one thread block for bitonic sort, which is performed entirely
// in shared memory. Has to be power of 2.
#define THREADS_PER_BITONIC_SORT 256
// How many elements are processed by one thread in bitonic sort kernel. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_BITONIC_SORT 4


/* --------------- BITONIC MERGE GLOBAL -------------- */

// How many threads are used per one thread block in GLOBAL bitonic merge. Has to be power of 2.
#define THREADS_PER_GLOBAL_MERGE 128
// How many elements are processed by one thread in GLOBAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_GLOBAL_MERGE 4


/* --------------- BITONIC MERGE LOCAL --------------- */

// How many threads are used per one thread block in LOCAL bitonic merge. Has to be power of 2.
#define THREADS_PER_LOCAL_MERGE 256
// How many elements are processed by one thread in LOCAL bitonic merge. Min value is 2.
// Has to be divisable by 2.
#define ELEMS_PER_THREAD_LOCAL_MERGE 4


/* ----------------- SAMPLE INDEXING ----------------- */

// Has to be greater or equal than NUM_SAMPLES. Has to be multiple of NUM_SAMPLES.
#define THREADS_PER_SAMPLE_INDEXING 128


/* ---------------- BUCKETS RELOCATION --------------- */

// How many threads are used per one thread block in kernel for buckets relocation. Has to be power of 2.
// Also has to be greater or equal than NUM_SAMPLES. Has to be multiple of NUM_SAMPLES.
#define THREADS_PER_BUCKETS_RELOCATION 128


/* --------- SEQUENTIAL ALGORITHM PARAMETERS --------- */

// How many splitters are used for buckets. From "N" splitters "N + 1" buckets are created.
#define NUM_SPLITTERS_SEQUENTIAL_KO 32
#define NUM_SPLITTERS_SEQUENTIAL_KV 32

// How many extra samples are taken for every splitter. Increases the queality of splitters (samples
// get sorted and only "NUM_SPLITTERS_SEQUENTIAL" splitters are taken from sorted array of samples).
#define OVERSAMPLING_FACTOR_KO 4
#define OVERSAMPLING_FACTOR_KV 4

// Threshold, when small sort is applied (in our case marge sort). Has to be greater or equeal than
// "NUM_SAMPLES_SEQUENTIAL".
#define SMALL_SORT_THRESHOLD_KO (1 << 15)
#define SMALL_SORT_THRESHOLD_KV (1 << 15)

#endif
