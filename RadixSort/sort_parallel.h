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

    // Device buffer for keys and values
    data_t *_d_keysBuffer = NULL, *_d_valuesBuffer = NULL;
    // Every radix digit has it's own corresponding bucket, where elements are scattered when sorted.
    // These vars hold bucket offsets (local and global) and global bucket sizes
    uint_t *_d_bucketOffsetsLocal = NULL, *_d_bucketOffsetsGlobal = NULL, *_d_bucketSizes = NULL;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortParallel::memoryAllocate(h_keys, h_values, arrayLength);

        uint_t elemsPerSortLocalMin = min(
            threadsSortLocalKo * elemsSortLocalKo, threadsSortLocalKv * elemsSortLocalKv
        );
        uint_t elemsPerSortLocalMax = max(
            threadsSortLocalKo * elemsSortLocalKo, threadsSortLocalKv * elemsSortLocalKv
        );
        uint_t radixMax = max(radixKo, radixKv);
        uint_t bucketsLen = radixMax * ((arrayLength - 1) / elemsPerSortLocalMin + 1);
        // In case table length not divisible by number of elements processed by one thread block in local radix
        // sort, data table is padded to the next multiple of number of elements per local radix sort.
        uint_t arrayLenRoundedUp = roundUp(arrayLength, elemsPerSortLocalMax);
        cudaError_t error;

        error = cudaMalloc((void **)&_d_keysBuffer, arrayLenRoundedUp * sizeof(*_d_keysBuffer));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_valuesBuffer, arrayLenRoundedUp * sizeof(*_d_valuesBuffer));
        checkCudaError(error);

        error = cudaMalloc((void **)&_d_bucketOffsetsLocal, bucketsLen * sizeof(*_d_bucketOffsetsLocal));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_bucketOffsetsGlobal, bucketsLen * sizeof(*_d_bucketOffsetsGlobal));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_bucketSizes, bucketsLen * sizeof(*_d_bucketSizes));
        checkCudaError(error);
    }

    /*
    Copies data from device to host. If sorting keys only, than "h_values" contains NULL.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        SortParallel::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        cudaError_t error;

        // TODO
    }

    /*
    Initializes library CUDPP, which implements scan() function
    */
    void cudppInitScan(CUDPPHandle *scanPlan, uint_t arrayLength)
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
        CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, arrayLength, 1, 0);

        if (result != CUDPP_SUCCESS)
        {
            printf("Error creating CUDPPPlan\n");
            getchar();
            exit(-1);
        }
    }

    /*
    Calls a kernel, which adds padding to array. If array length is shorter than "elemsPerThreadBlock", than
    padding is added to "elemsPerThreadBlock". I array length is greater than "elemsPerThreadBlock", then
    padding is added to the next power of 2 of array length.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void addPadding(data_t *d_keys, uint_t arrayLength)
    {
        uint_t elemsPerThreadBlock;

        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsSortLocalKo * elemsSortLocalKo;
        }
        else
        {
            elemsPerThreadBlock = threadsSortLocalKv * elemsSortLocalKv;
        }

        runAddPaddingKernel<sortOrder>(d_keys, arrayLength, roundUp(arrayLength, elemsPerThreadBlock));
    }

    /*
    Runs kernel, which sorts data blocks in shared memory with radix sort according to current radix diggit,
    which is specified with "bitOffset".
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runRadixSortLocalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t bitOffset)
    {
        uint_t elemsPerThreadBlock, sharedMemSize;

        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsSortLocalKo * elemsSortLocalKo;
            sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);
        }
        else
        {
            elemsPerThreadBlock = threadsSortLocalKv * elemsSortLocalKv;
            sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_keys);
        }

        dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsSortLocalKo : threadsSortLocalKv, 1, 1);

        if (sortingKeyOnly)
        {
            radixSortLocalKernel
                <threadsSortLocalKo, bitCountRadixKo, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, bitOffset
            );
        }
        else
        {
            radixSortLocalKernel
                <threadsSortLocalKv, bitCountRadixKv, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, bitOffset
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

        SortParallel::memoryDestroy();
        cudaError_t error;

        error = cudaFree(_d_keysBuffer);
        checkCudaError(error);
        error = cudaFree(_d_valuesBuffer);
        checkCudaError(error);

        error = cudaFree(_d_bucketOffsetsLocal);
        checkCudaError(error);
        error = cudaFree(_d_bucketOffsetsGlobal);
        checkCudaError(error);
        error = cudaFree(_d_bucketSizes);
        checkCudaError(error);
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
