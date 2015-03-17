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

    // Device buffer for keys and values
    data_t *_d_keysBuffer = NULL, *_d_valuesBuffer = NULL;
    // Holds ranks of all even and odd subblocks, that have to be merged
    uint_t *_d_ranksEven = NULL, *_d_ranksOdd = NULL;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);
        uint_t ranksLength = (arrayLenPower2 - 1) / min(subBlockSizeKo, subBlockSizeKv) + 1;
        cudaError_t error;

        SortParallel::memoryAllocate(h_keys, h_values, arrayLenPower2);

        error = cudaMalloc((void **)&_d_keysBuffer, arrayLenPower2 * sizeof(*_d_keysBuffer));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_valuesBuffer, arrayLenPower2 * sizeof(*_d_valuesBuffer));
        checkCudaError(error);

        error = cudaMalloc((void **)&_d_ranksEven, ranksLength * sizeof(*_d_ranksEven));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_ranksOdd, ranksLength * sizeof(*_d_ranksOdd));
        checkCudaError(error);
    }

    /*
    Method for destroying memory needed for sort. For sort testing purposes this method is public.
    */
    void memoryDestroy()
    {
        if (_arrayLength == 0)
        {
            return;
        }

        SortParallel::memoryDestroy();
        cudaError_t error;

        error = cudaFree(_d_keysBuffer);
        checkCudaError(error);
        error = cudaFree(_d_valuesBuffer);
        checkCudaError(error);

        error = cudaFree(_d_ranksEven);
        checkCudaError(error);
        error = cudaFree(_d_ranksOdd);
        checkCudaError(error);
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        //if (_sortOrder == ORDER_ASC)
        //{
        //    bitonicSortParallel<ORDER_ASC, true>(_d_keys, NULL, _arrayLength);
        //}
        //else
        //{
        //    bitonicSortParallel<ORDER_DESC, true>(_d_keys, NULL, _arrayLength);
        //}
    }

    /*
    wrapper for bitonic sort method.
    the code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        //if (_sortOrder == ORDER_ASC)
        //{
        //    bitonicSortParallel<ORDER_ASC, false>(_d_keys, _d_values, _arrayLength);
        //}
        //else
        //{
        //    bitonicSortParallel<ORDER_DESC, false>(_d_keys, _d_values, _arrayLength);
        //}
    }

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
