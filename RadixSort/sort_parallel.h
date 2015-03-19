#ifndef RADIX_SORT_PARALLEL_H
#define RADIX_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "../Utils/kernels_classes.h"
#include "../Utils/host.h"
#include "constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "kernels_common.h"
#include "kernels_key_only.h"
#include "kernels_key_value.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Parent class for parallel radix sort. Not to be used directly - it's inherited by bottom class, which performs
partial template specialization.
TODO implement for descending order.

Template params:
_Ko - Key-only
_Kv - Key-value
*/
template <
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsSortLocalKo, uint_t elemsSortLocalKo,
    uint_t threadsSortLocalKv, uint_t elemsSortLocalKv,
    uint_t threadsGenBucketsKo, uint_t threadsGenBucketsKv,
    uint_t threadsSortGlobalKo, uint_t threadsSortGlobalKv,
    uint_t bitCountRadixKo, uint_t radixKo,
    uint_t bitCountRadixKv, uint_t radixKv
>
class RadixSortParallelParent : public SortParallel, public AddPaddingBase<threadsPadding, elemsPadding>
{
protected:
    std::string _sortName = "Radix sort parallel";

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

/*
Base class for parallel merge sort.
Needed for template specialization.
*/
template<
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsSortLocalKo, uint_t elemsSortLocalKo,
    uint_t threadsSortLocalKv, uint_t elemsSortLocalKv,
    uint_t threadsGenBucketsKo, uint_t threadsGenBucketsKv,
    uint_t threadsSortGlobalKo, uint_t threadsSortGlobalKv,
    uint_t bitCountRadixKo, uint_t bitCountRadixKv
>
class RadixSortParallelBase : public RadixSortParallelParent<
    threadsPadding, elemsPadding,
    threadsSortLocalKo, elemsSortLocalKo,
    threadsSortLocalKv, elemsSortLocalKv,
    threadsGenBucketsKo, threadsGenBucketsKv,
    threadsSortGlobalKo, threadsSortGlobalKv,
    bitCountRadixKo, 1 << bitCountRadixKo,
    bitCountRadixKv, 1 << bitCountRadixKv
>
{};

/*
Class for parallel radix sort.
*/
class RadixSortParallel : public RadixSortParallelBase<
    THREADS_PER_PADDING, ELEMS_PER_THREAD_PADDING,
    THREADS_PER_LOCAL_SORT_KO, ELEMS_PER_THREAD_LOCAL_KO,
    THREADS_PER_LOCAL_SORT_KV, ELEMS_PER_THREAD_LOCAL_KV,
    THREADS_PER_GEN_BUCKETS_KO, THREADS_PER_GEN_BUCKETS_KV,
    THREADS_PER_GLOBAL_SORT_KO, THREADS_PER_GLOBAL_SORT_KV,
    BIT_COUNT_PARALLEL_KO, BIT_COUNT_PARALLEL_KV
>
{};

#endif
