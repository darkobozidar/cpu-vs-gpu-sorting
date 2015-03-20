#ifndef SAMPLE_SORT_PARALLEL_H
#define SAMPLE_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/kernels_classes.h"
#include "../Utils/host.h"
#include "../BitonicSort/sort_parallel.h"
#include "constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "kernels_common.h"
#include "kernels_key_only.h"
#include "kernels_key_value.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Base class for parallel sample sort.
Needed for template specialization.

Template params:
_Ko - Key-only
_Kv - Key-value
*/
template <
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsBitonicSortKo, uint_t elemsBitonicSortKo,
    uint_t threadsBitonicSortKv, uint_t elemsBitonicSortKv,
    uint_t threadsGlobalMergeKo, uint_t elemsGlobalMergeKo,
    uint_t threadsGlobalMergeKv, uint_t elemsGlobalMergeKv,
    uint_t threadsLocalMergeKo, uint_t elemsLocalMergeKo,
    uint_t threadsLocalMergeKv, uint_t elemsLocalMergeKv,
    uint_t threadsSampleIndexingKo, uint_t threadsSampleIndexingKv,
    uint_t threadsBucketRelocationKo, uint_t threadsBucketRelocationKv,
    uint_t numSamplesKo, uint_t numSamplesKv
>
class SampleSortParallelBase :
    public AddPaddingBase<threadsPadding, elemsPadding>,
    public BitonicSortParallelBase<
        threadsBitonicSortKo, elemsBitonicSortKo,
        threadsBitonicSortKv, elemsBitonicSortKv,
        threadsGlobalMergeKo, elemsGlobalMergeKo,
        threadsGlobalMergeKv, elemsGlobalMergeKv,
        threadsLocalMergeKo, elemsLocalMergeKo,
        threadsLocalMergeKv, elemsLocalMergeKv
    >
{
protected:
    std::string _sortName = "Sample sort parallel";

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};


/*
Class for parallel bitonic sort.
*/
class SampleSortParallel : public SampleSortParallelBase<
    THREADS_PADDING, ELEMS_PADDING,
    THREADS_BITONIC_SORT_KO, ELEMS_BITONIC_SORT_KO,
    THREADS_BITONIC_SORT_KV, ELEMS_BITONIC_SORT_KV,
    THREADS_GLOBAL_MERGE_KO, ELEMS_GLOBAL_MERGE_KO,
    THREADS_GLOBAL_MERGE_KV, ELEMS_GLOBAL_MERGE_KV,
    THREADS_LOCAL_MERGE_KO, ELEMS_LOCAL_MERGE_KO,
    THREADS_LOCAL_MERGE_KV, ELEMS_LOCAL_MERGE_KV,
    THREADS_SAMPLE_INDEXING_KO, THREADS_SAMPLE_INDEXING_KV,
    THREADS_BUCKETS_RELOCATION_KO, THREADS_BUCKETS_RELOCATION_KV,
    NUM_SAMPLES_PARALLEL_KO, NUM_SAMPLES_PARALLEL_KV
>
{};

#endif
