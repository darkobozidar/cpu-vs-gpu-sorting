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
    // Device buffer for keys and values
    data_t *_d_keysBuffer = NULL, *_d_valuesBuffer = NULL;
    // LOCAL samples:  NUM_SAMPLES samples collected from each data block sorted by initial bitonic sort
    // GLOBAL samples: NUM_SAMPLES samples collected from sorted LOCAL samples
    data_t *_d_samplesLocal = NULL, *_d_samplesGlobal = NULL;
    // Sizes and offsets of local (per every tile - thread block) buckets (gained after scan on bucket sizes)
    uint_t *_d_localBucketSizes = NULL, *_d_localBucketOffsets = NULL;
    // Offsets of entire (global) buckets, not just parts of buckets for every tile (local)
    uint_t *_h_globalBucketOffsets = NULL, *_d_globalBucketOffsets = NULL;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        // If array length is not multiple of number of elements processed by one thread block in initial
        // bitonic sort, than array is padded to that length.
        uint_t minElementsInitBitonicSort = min(
            threadsBitonicSortKo * elemsBitonicSortKo, threadsBitonicSortKv * elemsBitonicSortKv
        );
        uint_t maxElementsInitBitonicSort = max(
            threadsBitonicSortKo * elemsBitonicSortKo, threadsBitonicSortKv * elemsBitonicSortKv
        );
        uint_t maxNumSamples = max(numSamplesKo, numSamplesKv);

        uint_t arrayLenRoundedUp = roundUp(arrayLength, maxElementsInitBitonicSort);
        uint_t localSamplesDistance = (minElementsInitBitonicSort - 1) / maxNumSamples + 1;
        uint_t localSamplesLen = (arrayLenRoundedUp - 1) / localSamplesDistance + 1;
        // (number of all data blocks (tiles)) * (number buckets generated from "numSamples")
        uint_t localBucketsLen = ((arrayLenRoundedUp - 1) / minElementsInitBitonicSort + 1) * (maxNumSamples + 1);
        cudaError_t error;

        BitonicSortParallelBase::memoryAllocate(h_keys, h_values, arrayLenRoundedUp);

        /* HOST MEMORY */

        // Offsets of all global buckets (Needed for parallel sort)
        _h_globalBucketOffsets = (uint_t*)malloc((maxNumSamples + 1) * sizeof(*_h_globalBucketOffsets));
        checkMallocError(_h_globalBucketOffsets);

        /* DEVICE MEMORY */

        // Allocates keys and values
        error = cudaMalloc((void **)&_d_keysBuffer, arrayLenRoundedUp * sizeof(*_d_keysBuffer));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_valuesBuffer, arrayLenRoundedUp * sizeof(*_d_valuesBuffer));
        checkCudaError(error);

        // Arrays for storing samples
        error = cudaMalloc((void **)&_d_samplesLocal, localSamplesLen * sizeof(*_d_samplesLocal));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_samplesGlobal, maxNumSamples * sizeof(*_d_samplesGlobal));
        checkCudaError(error);

        // Arrays from bucket bookkeeping
        error = cudaMalloc((void **)&_d_localBucketSizes, localBucketsLen * sizeof(*_d_localBucketSizes));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_localBucketOffsets, localBucketsLen * sizeof(*_d_localBucketOffsets));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_globalBucketOffsets, (maxNumSamples + 1) * sizeof(*_d_globalBucketOffsets));
        checkCudaError(error);
    }

    /*
    Depending on the array length sorted array can be located in primary or in buffer array.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        bool sortingKeyOnly = h_values == NULL;
        uint_t threadsBitonicSort = sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv;
        uint_t elemsBitonicSort = sortingKeyOnly ? elemsBitonicSortKo : elemsBitonicSortKv;
        uint_t elemsInitBitonicSort = threadsBitonicSort * elemsBitonicSort;

        if (arrayLength <= elemsBitonicSort)
        {
            BitonicSortParallelBase::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        }
        else
        {
            cudaError_t error;
            // Copies keys
            error = cudaMemcpy(
                h_keys, (void *)_d_keysBuffer, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
            );
            checkCudaError(error);

            if (sortingKeyOnly)
            {
                return;
            }

            // Copies values
            error = cudaMemcpy(
                h_values, (void *)_d_valuesBuffer, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
            );
            checkCudaError(error);
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
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

        SortSequential::memoryDestroy();
        cudaError_t error;

        free(_h_globalBucketOffsets);

        error = cudaFree(_d_keysBuffer);
        checkCudaError(error);
        error = cudaFree(_d_valuesBuffer);
        checkCudaError(error);

        // Arrays for storing samples
        error = cudaFree(_d_samplesLocal);
        checkCudaError(error);
        error = cudaFree(_d_samplesGlobal);
        checkCudaError(error);

        // Arrays from bucket bookkeeping
        error = cudaFree(_d_localBucketSizes);
        checkCudaError(error);
        error = cudaFree(_d_localBucketOffsets);
        checkCudaError(error);
        error = cudaFree(_d_globalBucketOffsets);
        checkCudaError(error);
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
