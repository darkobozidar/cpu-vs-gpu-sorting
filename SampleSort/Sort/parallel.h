#ifndef SAMPLE_SORT_PARALLEL_H
#define SAMPLE_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_classes.h"
#include "../../Utils/host.h"
#include "../../BitonicSort/Sort/parallel.h"
#include "../constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "../Kernels/common.h"
#include "../Kernels/key_only.h"
#include "../Kernels/key_value.h"
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
                h_keys, (void *)_d_keysBuffer, _arrayLength * sizeof(*h_keys), cudaMemcpyDeviceToHost
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

    /*
    Initializes CUDPP scan.
    */
    void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen)
    {
        // Initializes the CUDPP Library
        CUDPPHandle theCudpp;
        cudppCreate(&theCudpp);

        CUDPPConfiguration config;
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_UINT;
        config.algorithm = CUDPP_SCAN;
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

        *scanPlan = 0;
        CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, tableLen, 1, 0);

        if (result != CUDPP_SUCCESS)
        {
            printf("Error creating CUDPPPlan for scan\n");
            getchar();
            exit(-1);
        }
    }

    /*
    Adds padding to input array.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void addPadding(data_t *d_keys, uint_t arrayLength)
    {
        uint_t threadsBitonicSort = sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv;
        uint_t elemsBitonicSort = sortingKeyOnly ? elemsBitonicSortKo : elemsBitonicSortKv;
        uint_t elemsInitBitonicSort = threadsBitonicSort * elemsBitonicSort;
        uint_t arrayLenRoundedUp = roundUp(arrayLength, elemsInitBitonicSort);

        runAddPaddingKernel<sortOrder>(d_keys, arrayLength, arrayLenRoundedUp);
    }

    /*
    Sorts sub-blocks of input data with NORMALIZED bitonic sort and collects NUM_SAMPLES_PARALLEL samples from every
    sorted chunk after the sort is complete.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitonicSortCollectSamplesKernel(data_t *d_keys, data_t *d_values, data_t *d_samples, uint_t arrayLength)
    {
        uint_t elemsPerThreadBlock, sharedMemSize;

        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsBitonicSortKo * elemsBitonicSortKo;
            sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);
        }
        else
        {
            elemsPerThreadBlock = threadsBitonicSortKv * elemsBitonicSortKv;
            sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_keys);
        }

        dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv, 1, 1);

        if (sortingKeyOnly)
        {
            bitonicSortCollectSamplesKernel
                <threadsBitonicSortKo, elemsBitonicSortKo, numSamplesKo, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_samples, arrayLength
            );
        }
        else
        {
            bitonicSortCollectSamplesKernel
                <threadsBitonicSortKv, elemsBitonicSortKv, numSamplesKv, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, d_samples, arrayLength
            );
        }
    }

    /*
    From sorted LOCAL samples collects (NUM_SAMPLES_PARALLEL) GLOBAL samples.
    */
    template <bool sortingKeyOnly>
    void runCollectGlobalSamplesKernel(data_t *d_samplesLocal, data_t *d_samplesGlobal, uint_t samplesLen)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? numSamplesKo : numSamplesKv, 1, 1);

        collectGlobalSamplesKernel<sortingKeyOnly ? numSamplesKo : numSamplesKv><<<dimGrid, dimBlock>>>(
            d_samplesLocal, d_samplesGlobal, samplesLen
        );
    }

    /*
    In all previously sorted (by initial bitonic sort) sub-blocks finds the indexes of all NUM_SAMPLES_PARALLEL
    global samples. From these indexes calculates the number of elements in each of the (NUM_SAMPLES_PARALLEL + 1)
    local buckets (calculates local bucket sizes) for every sorted sub-block.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runSampleIndexingKernel(
        data_t *d_keys, data_t *d_samples, uint_t *d_bucketSizes, uint_t arrayLength, uint_t numAllBuckets
    )
    {
        uint_t elemsPerBitonicSort, subBlocksPerThreadBlock;

        if (sortingKeyOnly)
        {
            elemsPerBitonicSort = threadsBitonicSortKo * elemsBitonicSortKo;
            subBlocksPerThreadBlock = threadsSampleIndexingKo / numSamplesKo;
        }
        else
        {
            elemsPerBitonicSort = threadsBitonicSortKv * elemsBitonicSortKv;
            subBlocksPerThreadBlock = threadsSampleIndexingKv / numSamplesKv;
        }

        // "Number of all sorted sub-blocks" / "number of sorted sub-blocks processed by one thread block"
        dim3 dimGrid((arrayLength / elemsPerBitonicSort - 1) / subBlocksPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsSampleIndexingKo : threadsSampleIndexingKv, 1, 1);

        if (sortingKeyOnly)
        {
            sampleIndexingKernel
                <threadsSampleIndexingKo, threadsBitonicSortKo, elemsBitonicSortKo, numSamplesKo, sortOrder>
                <<<dimGrid, dimBlock>>>(
                d_keys, d_samples, d_bucketSizes, arrayLength
            );
        }
        else
        {
            sampleIndexingKernel
                <threadsSampleIndexingKv, threadsBitonicSortKv, elemsBitonicSortKv, numSamplesKv, sortOrder>
                <<<dimGrid, dimBlock>>>(
                d_keys, d_samples, d_bucketSizes, arrayLength
            );
        }
    }

    /*
    With respect to local bucket sizes and offsets scatters elements to their global buckets. At the end it copies
    global bucket sizes (sizes of whole global buckets, not just bucket sizes per every sorted sub-block) to host.
    */
    template <bool sortingKeyOnly>
    void runBucketsRelocationKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, uint_t *h_globalBucketOffsets,
        uint_t *d_globalBucketOffsets, uint_t *d_localBucketSizes, uint_t *d_localBucketOffsets, uint_t arrayLength
    )
    {
        uint_t threadsBitonicSort = sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv;
        uint_t elemsBitonicSort = sortingKeyOnly ? elemsBitonicSortKo : elemsBitonicSortKv;
        uint_t numSamples = sortingKeyOnly ? numSamplesKo : numSamplesKv;

        uint_t elemsPerInitBitonicSort = threadsBitonicSort * elemsBitonicSort;
        // Shared mem size: For NUM_SAMPLES_PARALLEL samples (NUM_SAMPLES_PARALLEL + 1) buckets are created
        // "2" -> bucket sizes + bucket offsets
        uint_t sharedMemSize = 2 * (numSamples + 1) * sizeof(*d_localBucketSizes);

        dim3 dimGrid((arrayLength - 1) / elemsPerInitBitonicSort + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsBucketRelocationKo : threadsBucketRelocationKv, 1, 1);

        if (sortingKeyOnly)
        {
            bucketsRelocationKernel
                <threadsBucketRelocationKo, threadsBitonicSortKo, elemsBitonicSortKo, numSamplesKo, sortingKeyOnly>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_globalBucketOffsets, d_localBucketSizes,
                d_localBucketOffsets, arrayLength
            );
        }
        else
        {
            bucketsRelocationKernel
                <threadsBucketRelocationKv, threadsBitonicSortKv, elemsBitonicSortKv, numSamplesKv, sortingKeyOnly>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_globalBucketOffsets, d_localBucketSizes,
                d_localBucketOffsets, arrayLength
            );
        }

        cudaError_t error = cudaMemcpy(
            h_globalBucketOffsets, d_globalBucketOffsets, (numSamples + 1) * sizeof(*h_globalBucketOffsets),
            cudaMemcpyDeviceToHost
        );
        checkCudaError(error);
    }

    /*
    Sorts array with deterministic sample sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void sampleSortParallel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, data_t *d_samplesLocal,
        data_t *d_samplesGlobal, uint_t *h_globalBucketOffsets, uint_t *d_globalBucketOffsets,
        uint_t *d_localBucketSizes, uint_t *d_localBucketOffsets, uint_t arrayLength
    )
    {
        uint_t threadsBitonicSort = sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv;
        uint_t elemsBitonicSort = sortingKeyOnly ? elemsBitonicSortKo : elemsBitonicSortKv;
        uint_t numSamples = sortingKeyOnly ? numSamplesKo : numSamplesKv;

        uint_t elemsPerInitBitonicSort = threadsBitonicSort * elemsBitonicSort;
        // If table length is not multiple of number of elements processed by one thread block in initial
        // bitonic sort, than array is padded to that length.
        uint_t arrayLenRoundedUp = roundUp(arrayLength, elemsPerInitBitonicSort);
        uint_t localSamplesDistance = (elemsPerInitBitonicSort - 1) / numSamples + 1;
        uint_t localSamplesLen = (arrayLenRoundedUp - 1) / localSamplesDistance + 1;
        // (number of all data blocks (tiles)) * (number buckets generated from numSamples)
        uint_t localBucketsLen = ((arrayLenRoundedUp - 1) / elemsPerInitBitonicSort + 1) * (numSamples + 1);
        CUDPPHandle scanPlan;

        cudppInitScan(&scanPlan, localBucketsLen);
        addPadding<sortOrder, sortingKeyOnly>(d_keys, arrayLength);
        // Sorts sub-blocks of input data with bitonic sort and from every chunk collects numSamples samples
        runBitonicSortCollectSamplesKernel<sortOrder, sortingKeyOnly>(
            d_keys, d_values, d_samplesLocal, arrayLenRoundedUp
        );

        // Array has already been sorted
        if (arrayLength <= elemsPerInitBitonicSort)
        {
            return;
        }

        // Sorts collected local samples
        bitonicSortParallel<sortOrder, true>(d_samplesLocal, NULL, localSamplesLen);
        // From sorted LOCAL samples collects numSamples global samples
        runCollectGlobalSamplesKernel<sortingKeyOnly>(d_samplesLocal, d_samplesGlobal, localSamplesLen);
        // For all previously sorted sub-blocks calculates bucket sizes for global samples
        runSampleIndexingKernel<sortOrder, sortingKeyOnly>(
            d_keys, d_samplesGlobal, d_localBucketSizes, arrayLenRoundedUp, localBucketsLen
        );

        // Performs scan on local bucket sizes to gain local bucket offsets (global offset for all local buckets)
        CUDPPResult result = cudppScan(scanPlan, d_localBucketOffsets, d_localBucketSizes, localBucketsLen);
        if (result != CUDPP_SUCCESS)
        {
            printf("Error in cudppScan()\n");
            getchar();
            exit(-1);
        }

        // Moves elements to their corresponding global buckets and calculates global bucket offsets
        runBucketsRelocationKernel<sortingKeyOnly>(
            d_keys, d_values, d_keysBuffer, d_valuesBuffer, h_globalBucketOffsets, d_globalBucketOffsets,
            d_localBucketSizes, d_localBucketOffsets, arrayLength
        );

        // Sorts every bucket with bitonic sort
        uint_t previousOffset = 0;
        for (uint_t bucket = 0; bucket < numSamples + 1; bucket++)
        {
            // Padded part of the array doesn't need to be sorted in last bucket
            uint_t currentOffset = bucket < numSamples ? h_globalBucketOffsets[bucket] : arrayLength;
            uint_t bucketLen = currentOffset - previousOffset;

            if (bucketLen > 0)
            {
                bitonicSortParallel<sortOrder, true>(
                    d_keysBuffer + previousOffset, d_valuesBuffer + previousOffset, bucketLen
                );
            }
            previousOffset = currentOffset;
        }
    }

    /*
    Wrapper for sample sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            sampleSortParallel<ORDER_ASC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_samplesLocal, _d_samplesGlobal, _h_globalBucketOffsets,
                _d_globalBucketOffsets, _d_localBucketSizes, _d_localBucketOffsets, _arrayLength
            );
        }
        else
        {
            sampleSortParallel<ORDER_DESC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_samplesLocal, _d_samplesGlobal, _h_globalBucketOffsets,
                _d_globalBucketOffsets, _d_localBucketSizes, _d_localBucketOffsets, _arrayLength
            );
        }
    }

    /*
    Wrapper for sample sort method.
    The code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            sampleSortParallel<ORDER_ASC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_samplesLocal, _d_samplesGlobal,
                _h_globalBucketOffsets, _d_globalBucketOffsets, _d_localBucketSizes, _d_localBucketOffsets,
                _arrayLength
            );
        }
        else
        {
            sampleSortParallel<ORDER_DESC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_samplesLocal, _d_samplesGlobal,
                _h_globalBucketOffsets, _d_globalBucketOffsets, _d_localBucketSizes, _d_localBucketOffsets,
                _arrayLength
            );
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

        BitonicSortParallelBase::memoryDestroy();
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
