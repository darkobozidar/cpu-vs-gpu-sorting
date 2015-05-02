#ifndef BITONIC_SORT_ADAPTIVE_PARALLEL_H
#define BITONIC_SORT_ADAPTIVE_PARALLEL_H

#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/kernels_classes.h"
#include "../../Utils/cuda.h"
#include "../../Utils/host.h"
#include "../constants.h"
#include "../data_types.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "../Kernels/common.h"
#include "../Kernels/key_only.h"
#include "../Kernels/key_value.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Base class for parallel adaptive bitonic sort.
Needed for template specialization.

Template params:
_Ko - Key-only
_Kv - Key-value
*/
template <
    uint_t threadsBitonicSortKo, uint_t elemsBitonicSortKo,
    uint_t threadsBitonicSortKv, uint_t elemsBitonicSortKv,
    uint_t threadsLocalMergeKo, uint_t elemsLocalMergeKo,
    uint_t threadsLocalMergeKv, uint_t elemsLocalMergeKv,
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsInitIntervalsKo, uint_t elemsInitIntervalsKo,
    uint_t threadsInitIntervalsKv, uint_t elemsInitIntervalsKv,
    uint_t threadsGenIntervalsKo, uint_t elemsGenIntervalsKo,
    uint_t threadsGenIntervalsKv, uint_t elemsGenIntervalsKv
>
class BitonicSortAdaptiveParallelBase : public SortParallel, public AddPaddingBase<threadsPadding, elemsPadding>
{
protected:
    std::string _sortName = "Bitonic sort adaptive parallel";
    // Device buffer for keys and values
    data_t *_d_keysBuffer, *_d_valuesBuffer;
    // Stores intervals of bitonic subsequences
    interval_t *_d_intervals, *_d_intervalsBuffer;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);
        cudaError_t error;

        SortParallel::memoryAllocate(h_keys, h_values, arrayLenPower2);

        uint_t phasesAll = log2((double)arrayLenPower2);
        uint_t phasesBitonicMerge = log2((double)2 * min(threadsLocalMergeKo, threadsLocalMergeKv));
        uint_t intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

        // Allocates buffer for keys and values
        error = cudaMalloc((void **)&_d_keysBuffer, arrayLenPower2 * sizeof(*_d_keysBuffer));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_valuesBuffer, arrayLenPower2 * sizeof(*_d_valuesBuffer));
        checkCudaError(error);

        // Memory needed for storing intervals
        error = cudaMalloc((void **)&_d_intervals, intervalsLen * sizeof(*_d_intervals));
        checkCudaError(error);
        error = cudaMalloc((void **)&_d_intervalsBuffer, intervalsLen * sizeof(*_d_intervalsBuffer));
        checkCudaError(error);
    }

    /*
    Adds padding of MAX/MIN values to input table, depending if sort order is ascending or descending. This is
    needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
    */
    template <order_t sortOrder>
    void addPadding(data_t *d_keys, data_t *d_keysBuffer, uint_t arrayLength)
    {
        runAddPaddingKernel<sortOrder>(d_keys, d_keysBuffer, arrayLength, nextPowerOf2(arrayLength));
    }

    /*
    Sorts sub-blocks of input data with REGULAR bitonic sort (not NORMALIZED bitonic sort).
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitoicSortRegularKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength)
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

        // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
        // necessary for entire padded table to be ordered. It is only necessary that table is ordered to the next
        // multiple of number of elements processed by one thread block. This ensures that bitonic sequences get
        // created for entire original table length (padded elements are MIN/MAX values and sorting wouldn't change
        // anything).
        uint_t arrayLenRoundedUp;
        if (arrayLength > elemsPerThreadBlock)
        {
            arrayLenRoundedUp = roundUp(arrayLength, elemsPerThreadBlock);
        }
        // For sequences shorter than "arrayLenRoundedUp" only bitonic sort kernel is needed to sort them (without
        // any other kernels). In that case table size can be rounded to next power of 2.
        else
        {
            arrayLenRoundedUp = nextPowerOf2(arrayLength);
        }

        dim3 dimGrid((arrayLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsBitonicSortKo : threadsBitonicSortKv, 1, 1);

        if (sortingKeyOnly)
        {
            bitonicSortRegularKernel
                <threadsBitonicSortKo, elemsBitonicSortKo, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, arrayLenRoundedUp
            );
        }
        else
        {
            bitonicSortRegularKernel
                <threadsBitonicSortKv, elemsBitonicSortKv, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, arrayLenRoundedUp
            );
        }
    }

    /*
    Generates thread block size, number of thread blocks and size of shared memory for intervals kernel.
    */
    template <
        uint_t threadsIntervalsKo, uint_t elemsIntervalsKo, uint_t threadsIntervalsKv, uint_t elemsIntervalsKv,
        bool sortingKeyOnly
    >
    void generateKernelIntervalsParams(
        uint_t phasesAll, uint_t stepEnd, uint_t &threadBlockSize, uint_t &numThreadBlocks, uint_t &sharedMemSize
    )
    {
        uint_t intervalsLen = 1 << (phasesAll - stepEnd);

        if (sortingKeyOnly)
        {
            threadBlockSize = min((intervalsLen - 1) / elemsIntervalsKo + 1, threadsIntervalsKo);
            numThreadBlocks = (intervalsLen - 1) / (elemsIntervalsKo * threadBlockSize) + 1;
            // "2 *" because of BUFFER MEMORY for intervals
            sharedMemSize = 2 * elemsIntervalsKo * threadBlockSize * sizeof(interval_t);
        }
        else
        {
            threadBlockSize = min((intervalsLen - 1) / elemsIntervalsKv + 1, threadsIntervalsKv);
            numThreadBlocks = (intervalsLen - 1) / (elemsIntervalsKv * threadBlockSize) + 1;
            // "2 *" because of BUFFER MEMORY for intervals
            sharedMemSize = 2 * elemsIntervalsKv * threadBlockSize * sizeof(interval_t);
        }
    }

    /*
    Initializes intervals and continues to evolve them until the end step.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runInitIntervalsKernel(
        data_t *d_keys, interval_t *intervals, uint_t arrayLength, uint_t phasesAll, uint_t stepStart,
        uint_t stepEnd
    )
    {
        uint_t threadBlockSize, numThreadBlocks, sharedMemSize;

        generateKernelIntervalsParams
            <threadsInitIntervalsKo, elemsInitIntervalsKo, threadsInitIntervalsKv, elemsInitIntervalsKv,
            sortingKeyOnly>(
            phasesAll, stepEnd, threadBlockSize, numThreadBlocks, sharedMemSize
        );

        dim3 dimGrid(numThreadBlocks, 1, 1);
        dim3 dimBlock(threadBlockSize, 1, 1);

        if (sortingKeyOnly)
        {
            initIntervalsKernel<sortOrder, elemsInitIntervalsKo><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, intervals, arrayLength, stepStart, stepEnd
            );
        }
        else
        {
            initIntervalsKernel<sortOrder, elemsInitIntervalsKv><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, intervals, arrayLength, stepStart, stepEnd
            );
        }
    }

    /*
    Evolves intervals from start step to end step.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runGenerateIntervalsKernel(
        data_t *d_keys, interval_t *inputIntervals, interval_t *outputIntervals, uint_t arrayLength, uint_t phasesAll,
        uint_t phase, uint_t stepStart, uint_t stepEnd
    )
    {
        uint_t threadBlockSize, numThreadBlocks, sharedMemSize;

        generateKernelIntervalsParams
            <threadsGenIntervalsKo, elemsGenIntervalsKo, threadsGenIntervalsKv, elemsGenIntervalsKv,
            sortingKeyOnly>(
            phasesAll, stepEnd, threadBlockSize, numThreadBlocks, sharedMemSize
        );

        dim3 dimGrid(numThreadBlocks, 1, 1);
        dim3 dimBlock(threadBlockSize, 1, 1);

        if (sortingKeyOnly)
        {
            generateIntervalsKernel<sortOrder, elemsGenIntervalsKo><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, inputIntervals, outputIntervals, arrayLength, phase, stepStart, stepEnd
            );
        }
        else
        {
            generateIntervalsKernel<sortOrder, elemsGenIntervalsKv><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, inputIntervals, outputIntervals, arrayLength, phase, stepStart, stepEnd
            );
        }
    }

    /*
    Runs kernel, which performs bitonic merge from provided intervals.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitoicMergeIntervalsKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, interval_t *intervals,
        uint_t arrayLength, uint_t phase
    )
    {
        // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
        // necessary for entire padded table to be merged. It is only necessary that table is merged to the next
        // multiple of phase stride.
        uint_t arrayLenRoundedUp = roundUp(arrayLength, 1 << phase);
        uint_t elemsPerThreadBlock, sharedMemSize;

        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsLocalMergeKo * elemsLocalMergeKo;
            sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);
        }
        else
        {
            elemsPerThreadBlock = threadsLocalMergeKv * elemsLocalMergeKv;
            sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*d_keys);
        }

        dim3 dimGrid((arrayLenRoundedUp - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsLocalMergeKo : threadsLocalMergeKv, 1, 1);

        if (sortingKeyOnly)
        {
            bitonicMergeIntervalsKernel<threadsLocalMergeKo, elemsLocalMergeKo, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_keysBuffer, intervals, phase
            );
        }
        else
        {
            bitonicMergeIntervalsKernel<threadsLocalMergeKv, elemsLocalMergeKv, sortOrder>
                <<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, intervals, phase
            );
        }
    }

    /*
    Sorts data with parallel adaptive bitonic sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void bitonicSortAdaptiveParallel(
        data_t *&d_keys, data_t *&d_values, data_t *&d_keysBuffer, data_t *&d_valuesBuffer,
        interval_t *d_intervals, interval_t *d_intervalsBuffer, uint_t arrayLength
    )
    {
        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);
        uint_t elemsPerBlockBitonicSort, phasesBitonicMerge, phasesInitIntervals, phasesGenerateIntervals;

        if (sortingKeyOnly)
        {
            elemsPerBlockBitonicSort = threadsBitonicSortKo * elemsBitonicSortKo;
            phasesBitonicMerge = log2((double)(threadsLocalMergeKo * elemsLocalMergeKo));
            phasesInitIntervals = log2((double)threadsInitIntervalsKo * elemsInitIntervalsKo);
            phasesGenerateIntervals = log2((double)threadsGenIntervalsKo * elemsGenIntervalsKo);
        }
        else
        {
            elemsPerBlockBitonicSort = threadsBitonicSortKv * elemsBitonicSortKv;
            phasesBitonicMerge = log2((double)(threadsLocalMergeKv * elemsLocalMergeKv));
            phasesInitIntervals = log2((double)threadsInitIntervalsKv * elemsInitIntervalsKv);
            phasesGenerateIntervals = log2((double)threadsGenIntervalsKv * elemsGenIntervalsKv);
        }

        uint_t phasesAll = log2((double)arrayLenPower2);
        uint_t phasesBitonicSort = log2((double)min(arrayLenPower2, elemsPerBlockBitonicSort));

        if (phasesBitonicMerge < phasesBitonicSort)
        {
            printf(
                "\nNumber of phases executed in bitonic merge has to be lower than number of phases "
                "executed in initial bitonic sort. This is due to the fact, that regular bitonic sort is "
                "used (not normalized). This way the sort direction for entire thread block can be computed "
                "when executing bitonic merge, which is much more efficient.\n"
            );
            exit(EXIT_FAILURE);
        }

        addPadding<sortOrder>(d_keys, d_keysBuffer, arrayLength);
        runBitoicSortRegularKernel<sortOrder, sortingKeyOnly>(d_keys, d_values, arrayLength);

        for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
        {
            uint_t stepStart = phase;
            uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);

            if (phase > phasesBitonicMerge)
            {
                runInitIntervalsKernel<sortOrder, sortingKeyOnly>(
                    d_keys, d_intervals, arrayLenPower2, phasesAll, stepStart, stepEnd
                );

                // After initial intervals were generated intervals have to be evolved to the end step
                while (stepEnd > phasesBitonicMerge)
                {
                    interval_t *tempIntervals = d_intervals;
                    d_intervals = d_intervalsBuffer;
                    d_intervalsBuffer = tempIntervals;

                    stepStart = stepEnd;
                    stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
                    runGenerateIntervalsKernel<sortOrder, sortingKeyOnly>(
                        d_keys, d_intervalsBuffer, d_intervals, arrayLenPower2, phasesAll, phase, stepStart, stepEnd
                    );
                }
            }

            // Global merge with intervals
            runBitoicMergeIntervalsKernel<sortOrder, sortingKeyOnly>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_intervals, arrayLength, phase
            );

            // Exchanges keys
            data_t *tempTable = d_keys;
            d_keys = d_keysBuffer;
            d_keysBuffer = tempTable;

            if (!sortingKeyOnly)
            {
                // Exchanges values
                tempTable = d_values;
                d_values = d_valuesBuffer;
                d_valuesBuffer = tempTable;
            }
        }
    }

    /*
    Wrapper for adaptive bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortAdaptiveParallel<ORDER_ASC, true>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_intervals, _d_intervalsBuffer, _arrayLength
            );
        }
        else
        {
            bitonicSortAdaptiveParallel<ORDER_DESC, true>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_intervals, _d_intervalsBuffer, _arrayLength
            );
        }
    }

    /*
    Wrapper for adaptive bitonic sort method.
    The code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortAdaptiveParallel<ORDER_ASC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_intervals, _d_intervalsBuffer, _arrayLength
            );
        }
        else
        {
            bitonicSortAdaptiveParallel<ORDER_DESC, false>(
                _d_keys, _d_values, _d_keysBuffer, _d_valuesBuffer, _d_intervals, _d_intervalsBuffer, _arrayLength
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

        error = cudaFree(_d_intervals);
        checkCudaError(error);
        error = cudaFree(_d_intervalsBuffer);
        checkCudaError(error);
    }
};


/*
Class for parallel adaptive bitonic sort.
*/
class BitonicSortAdaptiveParallel : public BitonicSortAdaptiveParallelBase<
    THREADS_BITONIC_SORT_KO, ELEMS_BITONIC_SORT_KO,
    THREADS_BITONIC_SORT_KV, ELEMS_BITONIC_SORT_KV,
    THREADS_LOCAL_MERGE_KO, ELEMS_LOCAL_MERGE_KO,
    THREADS_LOCAL_MERGE_KV, ELEMS_LOCAL_MERGE_KV,
    THREADS_PADDING, ELEMS_PADDING,
    THREADS_INIT_INTERVALS_KO, ELEMS_INIT_INTERVALS_KO,
    THREADS_INIT_INTERVALS_KV, ELEMS_INIT_INTERVALS_KV,
    THREADS_GEN_INTERVALS_KO, ELEMS_GEN_INTERVALS_KO,
    THREADS_GEN_INTERVALS_KV, ELEMS_GEN_INTERVALS_KV
>
{};

#endif
