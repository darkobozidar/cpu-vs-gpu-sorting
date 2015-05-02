#ifndef MERGE_SORT_PARALLEL_H
#define MERGE_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    uint_t threadsGenRanksKo, uint_t threadsGenRanksKv
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
        uint_t elemsPerThreadBlock = max(
            threadsMergeSortKo * elemsMergeSortKo, threadsMergeSortKv * elemsMergeSortKv
        );
        uint_t arrayLenRoundedUp = max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
        uint_t ranksLength = (arrayLenRoundedUp - 1) / min(subBlockSizeKo, subBlockSizeKv) + 1;
        cudaError_t error;

        SortParallel::memoryAllocate(h_keys, h_values, arrayLenRoundedUp);

        error = cudaMalloc((void **)&_d_keysBuffer, arrayLenRoundedUp * sizeof(*_d_keysBuffer));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_valuesBuffer, arrayLenRoundedUp * sizeof(*_d_valuesBuffer));
        checkCudaError(error);

        error = cudaMalloc((void **)&_d_ranksEven, ranksLength * sizeof(*_d_ranksEven));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_ranksOdd, ranksLength * sizeof(*_d_ranksOdd));
        checkCudaError(error);
    }

    /*
    In case if number of phases in global merge is even, than merged array is locaded in primary array, else
    it is located in buffer array.
    Generally this memory copy wouldn't be needed and the same result could be achieved with, if the references
    to memory were passed by reference, but this way sort works little faster.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        uint_t elemsPerInitMergeSort;

        if (h_values == NULL)
        {
            elemsPerInitMergeSort = threadsMergeSortKo * elemsMergeSortKo;
        }
        else
        {
            elemsPerInitMergeSort = threadsMergeSortKv * elemsMergeSortKv;
        }

        uint_t arrayLenRoundedUp = max(nextPowerOf2(arrayLength), elemsPerInitMergeSort);
        uint_t numMergePhases = log2((double)arrayLenRoundedUp) - log2((double)elemsPerInitMergeSort);

        if (numMergePhases % 2 == 0)
        {
            SortParallel::memoryCopyAfterSort(h_keys, h_values, arrayLength);
        }
        else
        {
            cudaError_t error;

            // Copies keys
            error = cudaMemcpy(
                h_keys, (void *)_d_keysBuffer, _arrayLength * sizeof(*h_keys), cudaMemcpyDeviceToHost
            );
            checkCudaError(error);

            if (h_values == NULL)
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
    Returns the size of array that needs to be merged. If array size is power of 2, than array size is returned.
    In opposite case array size is broken into 2 parts:
    - main part (previous power of 2 of table length)
    - remainder (table length - main part size)

    > Remainder needs to be merged only until it is sorted (function returns size of whole array rounded up).
    > After that only main part of the array has to be merged (function returns size of "main part").
    > In last merge phase "main part" and "remainder" have to be merged.
    */
    uint_t calculateMergeArraySize(uint_t arrayLength, uint_t sortedBlockSize)
    {
        uint_t arrayLenMerge = previousPowerOf2(arrayLength);
        uint_t mergedBlockSize = 2 * sortedBlockSize;

        // Array length is already a power of 2
        if (arrayLenMerge != arrayLength)
        {
            // Number of elements over the power of 2 length
            uint_t remainder = arrayLength - arrayLenMerge;

            // Remainder needs to be merged only if it is GREATER or EQUAL than current sorted block size. If it is
            // SMALLER than current sorted block size, this means that it has already been sorted. In that case only
            // the main part of array (previous power of 2) has to be merged.
            if (remainder >= sortedBlockSize)
            {
                arrayLenMerge += roundUp(remainder, 2 * sortedBlockSize);
            }
            // In last merge phase the whole array has to be merged (main part of array + remainder)
            else if (arrayLenMerge == sortedBlockSize)
            {
                arrayLenMerge += sortedBlockSize;
            }
        }

        return arrayLenMerge;
    }

    /*
    Calls a kernel, which adds padding to array. If array length is shorter than "elemsPerThreadBlock", than
    padding is added to "elemsPerThreadBlock". I array length is greater than "elemsPerThreadBlock", then
    padding is added to the next power of 2 of array length.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void addPadding(data_t *d_keys, data_t *d_keysBuffer, uint_t arrayLength)
    {
        uint_t threadsMergeSort = sortingKeyOnly ? threadsMergeSortKo : threadsMergeSortKv;
        uint_t elemsMergeSort = sortingKeyOnly ? elemsMergeSortKo : elemsMergeSortKv;
        uint_t elemsPerThreadBlock = threadsMergeSort * elemsMergeSort;
        uint_t arrayLenRoundedUp = max(nextPowerOf2(arrayLength), elemsPerThreadBlock);
        runAddPaddingKernel<sortOrder>(d_keys, d_keysBuffer, arrayLength, arrayLenRoundedUp);
    }

    /*
    Sorts sub-blocks of data with merge sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runMergeSortKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength)
    {
        uint_t elemsPerThreadBlock, sharedMemSize;

        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsMergeSortKo * elemsMergeSortKo;
            // "2 *" because buffer shared memory is used in kernel alongside primary shared memory
            sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_keys);
        }
        else
        {
            elemsPerThreadBlock = threadsMergeSortKv * elemsMergeSortKv;
            // "2 *" because buffer shared memory is used in kernel alongside primary shared memory
            // "2 *" for because values are sorted alongside keys
            sharedMemSize = 4 * elemsPerThreadBlock * sizeof(*d_keys);
        }

        // In case array length is not power of 2, array is padded with MIN/MAX values to the next power of 2.
        // Padded MIN/MAX elements don't need to be sorted (they are already "sorted"), that's why array is
        // sorted only to the next multiply of number of elements processed by one thread block.
        uint_t arrayLenRoundedUp = roundUp(arrayLength, elemsPerThreadBlock);
        dim3 dimGrid((arrayLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsMergeSortKo : threadsMergeSortKv, 1, 1);

        if (sortingKeyOnly)
        {
            mergeSortKernel<threadsMergeSortKo, elemsMergeSortKo, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys
            );
        }
        else
        {
            mergeSortKernel<threadsMergeSortKv, elemsMergeSortKv, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values
            );
        }
    }

    /*
    If array length is not power of 2, then array is split into 2 parts:
    - main part (previous power of 2 of array length)
    - remainder (array length - main part size)

    Remainder needs to be merged only until it is sorted. After that only main part of the array has to be merged.
    For every merge phase data is passed from primary array to buffer (and vice versa)
    In last merge phase "main part" and "remainder" have to be merged. If number of merge phases is odd, than
    "remainder" is located in buffer, while "main part" is located in primary array. In that case "remainder" has
    to be copied to "main array", so they can be merged.
    */
    template <bool sortingKeyOnly>
    uint_t copyPaddedElements(
        data_t *d_keysFrom, data_t *d_valuesFrom, data_t *d_keysTo, data_t *d_valuesTo, uint_t arrayLength,
        uint_t sortedBlockSize, uint_t lastPaddingMergePhase
    )
    {
        uint_t arrayLenMerge = previousPowerOf2(arrayLength);
        uint_t remainder = arrayLength - arrayLenMerge;

        // If remainder has to be merged || if this is last merge phase (main part and remainder have to be merged)
        if (remainder >= sortedBlockSize || arrayLenMerge == sortedBlockSize)
        {
            // Calculates current merge phase
            uint_t currentMergePhase = log2((double)(2 * sortedBlockSize));
            uint_t phaseDifference = currentMergePhase - lastPaddingMergePhase;

            // If difference between phase when remainder was last merged and current phase is EVEN, this means
            // that remainder is located in buffer while main part is located in primary array. In that case
            // remainder is copied into primary array.
            if (phaseDifference % 2 == 0)
            {
                cudaError_t error;

                error = cudaMemcpy(d_keysTo, d_keysFrom, remainder * sizeof(*d_keysTo), cudaMemcpyDeviceToDevice);
                checkCudaError(error);

                if (!sortingKeyOnly)
                {
                    error = cudaMemcpy(
                        d_valuesTo, d_valuesFrom, remainder * sizeof(*d_valuesTo), cudaMemcpyDeviceToDevice
                    );
                    checkCudaError(error);
                }
            }

            // Saves last phase when remainder was merged.
            lastPaddingMergePhase = currentMergePhase;
        }

        return lastPaddingMergePhase;
    }

    /*
    Generates array of ranks/boundaries of sub-block, which will be merged.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runGenerateRanksKernel(
        data_t *d_keys, uint_t *d_ranksEven, uint_t *d_ranksOdd, uint_t arrayLength, uint_t sortedBlockSize
    )
    {
        uint_t subBlockSize = sortingKeyOnly ? subBlockSizeKo : subBlockSizeKv;
        uint_t threadsKernel = sortingKeyOnly ? threadsGenRanksKo : threadsGenRanksKv;

        uint_t arrayLenRoundedUp = calculateMergeArraySize(arrayLength, sortedBlockSize);
        uint_t numAllSamples = (arrayLenRoundedUp - 1) / subBlockSize + 1;
        uint_t threadBlockSize = min(numAllSamples, threadsKernel);

        dim3 dimGrid((numAllSamples - 1) / threadBlockSize + 1, 1, 1);
        dim3 dimBlock(threadBlockSize, 1, 1);

        if (sortingKeyOnly)
        {
            generateRanksKernel<subBlockSizeKo, sortOrder><<<dimGrid, dimBlock>>>(
                d_keys, d_ranksEven, d_ranksOdd, sortedBlockSize
            );
        }
        else
        {
            generateRanksKernel<subBlockSizeKv, sortOrder><<<dimGrid, dimBlock>>>(
                d_keys, d_ranksEven, d_ranksOdd, sortedBlockSize
            );
        }
    }

    /*
    Executes merge kernel, which merges all consecutive sorted blocks.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runMergeKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, uint_t *d_ranksEven,
        uint_t *d_ranksOdd, uint_t arrayLength, uint_t sortedBlockSize
    )
    {
        uint_t arrayLenMerge = calculateMergeArraySize(arrayLength, sortedBlockSize);
        uint_t mergedBlockSize = 2 * sortedBlockSize;
        // Number of merged blocks
        uint_t numMergedBlocks = (arrayLenMerge - 1) / mergedBlockSize + 1;
        uint_t subBlockSize = sortingKeyOnly ? subBlockSizeKo : subBlockSizeKv;
        // Sub-blocks of size "subBlockSize" per one merged block
        uint_t subBlocksPerMergedBlock = (mergedBlockSize - 1) / subBlockSize + 1;

        // "+ 1" is used because 1 thread block more is needed than number of samples/ranks/splitters per
        // merged block (eg: if we cut something n times, we get n + 1 pieces)
        dim3 dimGrid(subBlocksPerMergedBlock + 1, numMergedBlocks, 1);
        dim3 dimBlock(subBlockSize, 1, 1);

        if (sortingKeyOnly)
        {
            mergeKernel<subBlockSizeKo, sortOrder><<<dimGrid, dimBlock>>>(
                d_keys, d_keysBuffer, d_ranksEven, d_ranksOdd, sortedBlockSize
            );
        }
        else
        {
            mergeKernel<subBlockSizeKv, sortOrder><<<dimGrid, dimBlock>>>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_ranksEven, d_ranksOdd, sortedBlockSize
            );
        }
    }

    /*
    Sorts data with parallel merge sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void mergeSortParallel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, uint_t *d_ranksEven,
        uint_t *d_ranksOdd, uint_t arrayLength
    )
    {
        uint_t sortedBlockSize;

        if (sortingKeyOnly)
        {
            sortedBlockSize = threadsMergeSortKo * elemsMergeSortKo;
        }
        else
        {
            sortedBlockSize = threadsMergeSortKv * elemsMergeSortKv;
        }
        // Last merge phase when "remainder" was merged (part of array over it's last power of 2 in length)
        uint_t lastPaddingMergePhase = log2((double)(sortedBlockSize));
        uint_t arrayLenPrevPower2 = previousPowerOf2(arrayLength);

        addPadding<sortOrder, sortingKeyOnly>(d_keys, d_keysBuffer, arrayLength);
        runMergeSortKernel<sortOrder, sortingKeyOnly>(d_keys, d_values, arrayLength);

        while (sortedBlockSize < arrayLength)
        {
            // Exchanges keys and values
            data_t* temp = d_keys;
            d_keys = d_keysBuffer;
            d_keysBuffer = temp;

            if (!sortingKeyOnly)
            {
                temp = d_values;
                d_values = d_valuesBuffer;
                d_valuesBuffer = temp;
            }

            lastPaddingMergePhase = copyPaddedElements<sortingKeyOnly>(
                d_keys + arrayLenPrevPower2, d_values + arrayLenPrevPower2,
                d_keysBuffer + arrayLenPrevPower2, d_valuesBuffer + arrayLenPrevPower2, arrayLength,
                sortedBlockSize, lastPaddingMergePhase
            );
            runGenerateRanksKernel<sortOrder, sortingKeyOnly>(
                d_keysBuffer, d_ranksEven, d_ranksOdd, arrayLength, sortedBlockSize
            );
            runMergeKernel<sortOrder, sortingKeyOnly>(
                d_keysBuffer, d_valuesBuffer, d_keys, d_values, d_ranksEven, d_ranksOdd, arrayLength,
                sortedBlockSize
            );

            sortedBlockSize *= 2;
        }
    }

    /*
    Wrapper for merge sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            mergeSortParallel<ORDER_ASC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_ranksEven, _d_ranksOdd, _arrayLength
            );
        }
        else
        {
            mergeSortParallel<ORDER_DESC, true>(
                _d_keys, NULL, _d_keysBuffer, NULL, _d_ranksEven, _d_ranksOdd, _arrayLength
            );
        }
    }

    /*
    Wrapper for merge sort method.
    The code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            mergeSortParallel<ORDER_ASC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_ranksEven, _d_ranksOdd, _arrayLength
            );
        }
        else
        {
            mergeSortParallel<ORDER_DESC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_ranksEven, _d_ranksOdd, _arrayLength
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

        error = cudaFree(_d_ranksEven);
        checkCudaError(error);
        error = cudaFree(_d_ranksOdd);
        checkCudaError(error);
    }
};


/*
Class for parallel merge sort.
*/
class MergeSortParallel : public MergeSortParallelBase<
    SUB_BLOCK_SIZE_KO, SUB_BLOCK_SIZE_KV,
    THREADS_PADDING, ELEMS_PADDING,
    THREADS_MERGE_SORT_KO, ELEMS_MERGE_SORT_KO,
    THREADS_MERGE_SORT_KV, ELEMS_MERGE_SORT_KV,
    THREADS_GEN_RANKS_KO, THREADS_GEN_RANKS_KV
>
{};

#endif
