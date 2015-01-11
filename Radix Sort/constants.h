#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ------------------ PADDING KERNEL ----------------- */

// How many threads are used per on thread block for padding. Has to be power of 2.
#define THREADS_PER_PADDING 128
// How many table elements are processed by one thread in padding kernel. Min value is 2.
#define ELEMS_PER_THREAD_PADDING 4

#define THREADS_PER_LOCAL_SORT 128
#define ELEMS_PER_THREAD_LOCAL 4

#define THREADS_PER_GLOBAL_SORT 128

#define BIT_COUNT_PARALLEL 4
#define RADIX_PARALLEL (1 << BIT_COUNT_PARALLEL)
#define RADIX_MASK_PARALLEL ((1 << BIT_COUNT_PARALLEL) - 1)

#define BIT_COUNT_SEQUENTIAL 8
#define RADIX_SEQUENTIAL (1 << BIT_COUNT_SEQUENTIAL)
#define RADIX_MASK_SEQUENTIAL ((1 << BIT_COUNT_SEQUENTIAL) - 1)

#endif
