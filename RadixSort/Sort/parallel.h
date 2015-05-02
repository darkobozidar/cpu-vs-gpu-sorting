#ifndef RADIX_SORT_PARALLEL_H
#define RADIX_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/kernels_classes.h"
#include "../../Utils/host.h"
#include "../constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "../Kernels/common.h"
#include "../Kernels/key_only.h"
#include "../Kernels/key_value.h"
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
    Depending of the number of phases performed by radix sort the sorted array can be located in primary
    or buffer array.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        bool sortingKeyOnly = h_values == NULL;
        uint_t bitCountRadix = sortingKeyOnly ? bitCountRadixKo : bitCountRadixKv;
        uint_t numPhases = DATA_TYPE_BITS / bitCountRadix;

        if (numPhases % 2 == 0)
        {
            SortParallel::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        }
        else
        {
            // Counting sort was performed
            cudaError_t error = cudaMemcpy(
                h_keys, (void *)_d_keysBuffer, _arrayLength * sizeof(*_h_keys), cudaMemcpyDeviceToHost
            );
            checkCudaError(error);

            if (h_values == NULL)
            {
                return;
            }

            error = cudaMemcpy(
                h_values, (void *)_d_valuesBuffer, arrayLength * sizeof(*h_values), cudaMemcpyDeviceToHost
            );
            checkCudaError(error);
        }
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
    Runs kernel, which sorts data blocks in shared memory with radix sort according to current radix digit,
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

    /*
    Runs kernel, which generates local bucket offsets and sizes.
    */
    template <bool sortingKeyOnly>
    void runGenerateBucketsKernel(
        data_t *d_keys, uint_t *blockOffsets, uint_t *blockSizes, uint_t arrayLength, uint_t bitOffset
    )
    {
        uint_t radix = sortingKeyOnly ? radixKo : radixKv;
        uint_t elemsPerLocalSort;

        if (sortingKeyOnly)
        {
            elemsPerLocalSort = threadsSortLocalKo * elemsSortLocalKo;
        }
        else
        {
            elemsPerLocalSort = threadsSortLocalKv * elemsSortLocalKv;
        }

        // Shared memory size:
        // - "elemsPerLocalSort" -> container for elements read from global memory into shared memory
        // - "2 * radix"         -> bucket local sizes + bucket local offsets
        uint_t sharedMemSize = elemsPerLocalSort * sizeof(uint_t) + 2 * radix * sizeof(uint_t);

        dim3 dimGrid((arrayLength - 1) / elemsPerLocalSort + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsGenBucketsKo : threadsGenBucketsKv, 1, 1);

        if (sortingKeyOnly)
        {
            generateBucketsKernel
                <threadsGenBucketsKo, threadsSortLocalKo, elemsSortLocalKo, radixKo>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, blockOffsets, blockSizes, bitOffset
            );
        }
        else
        {
            generateBucketsKernel
                <threadsGenBucketsKv, threadsSortLocalKv, elemsSortLocalKv, radixKv>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, blockOffsets, blockSizes, bitOffset
            );
        }
    }

    /*
    Scatters elements to their corresponding buckets according to current radix digit, which is specified
    with "bitOffset".
    */
    template <bool sortingKeyOnly>
    void runRadixSortGlobalKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, uint_t *offsetsLocal,
        uint_t *offsetsGlobal, uint_t arrayLength, uint_t bitOffset
    )
    {
        uint_t elemsPerLocalSort, sharedMemSize;

        if (sortingKeyOnly)
        {
            elemsPerLocalSort = threadsSortLocalKo * elemsSortLocalKo;
            sharedMemSize = elemsPerLocalSort * sizeof(*d_keys);
        }
        else
        {
            elemsPerLocalSort = threadsSortLocalKv * elemsSortLocalKv;
            sharedMemSize = 2 * elemsPerLocalSort * sizeof(*d_keys);
        }

        dim3 dimGrid((arrayLength - 1) / elemsPerLocalSort + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsSortGlobalKo : threadsSortGlobalKv, 1, 1);

        if (sortingKeyOnly)
        {
            radixSortGlobalKernel
                <threadsSortGlobalKo, threadsSortLocalKo, elemsSortLocalKo, radixKo>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_keysBuffer, offsetsLocal, offsetsGlobal, bitOffset
            );
        }
        else
        {
            radixSortGlobalKernel
                <threadsSortGlobalKv, threadsSortLocalKv, elemsSortLocalKv, radixKv>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, offsetsLocal, offsetsGlobal, bitOffset
            );
        }
    }

    /*
    Sorts data with parallel radix sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void radixSortParallel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer,
        uint_t *d_bucketOffsetsLocal, uint_t *d_bucketOffsetsGlobal, uint_t *d_bucketSizes, uint_t arrayLength
    )
    {
        uint_t radix = sortingKeyOnly ? radixKo : radixKv;
        uint_t bitCountRadix = sortingKeyOnly ? bitCountRadixKo : bitCountRadixKv;
        uint_t elemsPerLocalSort;

        if (sortingKeyOnly)
        {
            elemsPerLocalSort = threadsSortLocalKo * elemsSortLocalKo;
        }
        else
        {
            elemsPerLocalSort = threadsSortLocalKv * elemsSortLocalKv;
        }

        uint_t bucketsLen = radix * ((arrayLength - 1) / elemsPerLocalSort + 1);
        CUDPPHandle scanPlan;

        cudppInitScan(&scanPlan, bucketsLen);
        addPadding<sortOrder, sortingKeyOnly>(d_keys, arrayLength);

        for (uint_t bitOffset = 0; bitOffset < sizeof(data_t) * 8; bitOffset += bitCountRadix)
        {
            runRadixSortLocalKernel<sortOrder, sortingKeyOnly>(d_keys, d_values, arrayLength, bitOffset);
            runGenerateBucketsKernel<sortingKeyOnly>(
                d_keys, d_bucketOffsetsLocal, d_bucketSizes, arrayLength, bitOffset
            );

            // Performs global scan in order to calculate global bucket offsets from local bucket sizes
            CUDPPResult result = cudppScan(scanPlan, d_bucketOffsetsGlobal, d_bucketSizes, bucketsLen);
            if (result != CUDPP_SUCCESS)
            {
                printf("Error in cudppScan()\n");
                getchar();
                exit(-1);
            }

            runRadixSortGlobalKernel<sortingKeyOnly>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_bucketOffsetsLocal, d_bucketOffsetsGlobal,
                arrayLength, bitOffset
            );

            data_t *temp = d_keys;
            d_keys = d_keysBuffer;
            d_keysBuffer = temp;

            if (!sortingKeyOnly)
            {
                temp = d_values;
                d_values = d_valuesBuffer;
                d_valuesBuffer = temp;
            }
        }
    }

    /*
    Wrapper for radix sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            radixSortParallel<ORDER_ASC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_bucketOffsetsLocal, _d_bucketOffsetsGlobal, _d_bucketSizes,
                _arrayLength
            );
        }
        else
        {
            radixSortParallel<ORDER_DESC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_bucketOffsetsLocal, _d_bucketOffsetsGlobal, _d_bucketSizes,
                _arrayLength
            );
        }
    }

    /*
    wrapper for radix sort method.
    the code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            radixSortParallel<ORDER_ASC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_bucketOffsetsLocal, _d_bucketOffsetsGlobal,
                _d_bucketSizes, _arrayLength
            );
        }
        else
        {
            radixSortParallel<ORDER_ASC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_bucketOffsetsLocal, _d_bucketOffsetsGlobal,
                _d_bucketSizes, _arrayLength
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
    THREADS_PADDING, ELEMS_PADDING,
    THREADS_LOCAL_SORT_KO, ELEMS_LOCAL_KO,
    THREADS_LOCAL_SORT_KV, ELEMS_LOCAL_KV,
    THREADS_GEN_BUCKETS_KO, THREADS_GEN_BUCKETS_KV,
    THREADS_GLOBAL_SORT_KO, THREADS_GLOBAL_SORT_KV,
    BIT_COUNT_PARALLEL_KO, BIT_COUNT_PARALLEL_KV
>
{};

#endif
