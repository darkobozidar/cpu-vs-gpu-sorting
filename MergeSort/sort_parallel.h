#ifndef MERGE_SORT_PARALLEL_H
#define MERGE_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
Base class for parallel merge sort.
Needed for template specialization.

Template params:
_Ko - Key-only
_Kv - Key-value
*/
template <
    uint_t subBlockSizeKo, uint_t subBlockSizeKv,
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsMergeSortKo, uint_t elemsMergeSortKo,
    uint_t threadsMergeSortKv, uint_t elemsMergeSortKv,
    uint_t threadsGenRanksKo, uint_t elemsGenRanksKo
>
class MergeSortParallelBase : public SortParallel, public AddPaddingBase<threadsPadding, elemsPadding>
{
protected:
    std::string _sortName = "Merge sort parallel";

    ///*
    //Wrapper for bitonic sort method.
    //The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    //*/
    //void sortKeyOnly()
    //{
    //    if (_sortOrder == ORDER_ASC)
    //    {
    //        bitonicSortParallel<ORDER_ASC, true>(_d_keys, NULL, _arrayLength);
    //    }
    //    else
    //    {
    //        bitonicSortParallel<ORDER_DESC, true>(_d_keys, NULL, _arrayLength);
    //    }
    //}

    ///*
    //wrapper for bitonic sort method.
    //the code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    //*/
    //void sortKeyValue()
    //{
    //    if (_sortOrder == ORDER_ASC)
    //    {
    //        bitonicSortParallel<ORDER_ASC, false>(_d_keys, _d_values, _arrayLength);
    //    }
    //    else
    //    {
    //        bitonicSortParallel<ORDER_DESC, false>(_d_keys, _d_values, _arrayLength);
    //    }
    //}

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};


/*
Class for parallel merge sort.
*/
class MergeSortParallel : public MergeSortParallelBase<
    SUB_BLOCK_SIZE_KO, SUB_BLOCK_SIZE_KV,
    THREADS_PER_PADDING, ELEMS_PER_THREAD_PADDING,
    THREADS_PER_MERGE_SORT_KO, ELEMS_PER_THREAD_MERGE_SORT_KO,
    THREADS_PER_MERGE_SORT_KV, ELEMS_PER_THREAD_MERGE_SORT_KV,
    THREADS_PER_GEN_RANKS_KO, THREADS_PER_GEN_RANKS_KV
>
{};

#endif
