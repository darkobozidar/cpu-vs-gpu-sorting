#ifndef BITONIC_SORT_ADAPTIVE_PARALLEL_H
#define BITONIC_SORT_ADAPTIVE_PARALLEL_H

#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "data_types.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "kernels_common.h"
#include "kernels_key_only.h"
#include "kernels_key_value.h"
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
    uint_t threadsGlobalMergeKo, uint_t elemsGlobalMergeKo,
    uint_t threadsGlobalMergeKv, uint_t elemsGlobalMergeKv,
    uint_t threadsLocalMergeKo, uint_t elemsLocalMergeKo,
    uint_t threadsLocalMergeKv, uint_t elemsLocalMergeKv,
    uint_t threadsPadding, uint_t elemsPadding,
    uint_t threadsInitIntervalsKo, uint_t elemsInitIntervalsKo,
    uint_t threadsInitIntervalsKv, uint_t elemsInitIntervalsKv,
    uint_t threadsGenIntervalsKo, uint_t elemsGenIntervalsKo,
    uint_t threadsGenIntervalsKv, uint_t elemsGenintervalsKv
>
class BitonicSortAdaptiveParallelBase : public BitonicSortParallelBase<
    threadsBitonicSortKo, elemsBitonicSortKo, threadsBitonicSortKv, elemsBitonicSortKv,
    threadsGlobalMergeKo, elemsGlobalMergeKo, threadsGlobalMergeKv, elemsGlobalMergeKv,
    threadsLocalMergeKo, elemsLocalMergeKo, threadsLocalMergeKv, elemsLocalMergeKv
>
{
protected:
    std::string _sortName = "Bitonic sort multistep parallel";
    // Device buffer for keys and values
    data_t *_d_keysBuffer, *_k_valuesBuffer;
    // Stores intervals of bitonic subsequences
    interval_t *_d_intervals, *_d_intervalsBuffer;

    /*
    Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
    needed, if table length is not power of 2. In order for bitonic sort to work, table length has to be power of 2.
    */
    template <order_t sortOrder>
    void runAddPaddingKernel(data_t *d_keys, data_t *d_keysBuffer, uint_t arrayLength)
    {
        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);

        // If table length is already power of 2, than no padding is needed
        if (arrayLength == arrayLenPower2)
        {
            return;
        }

        uint_t paddingLength = arrayLenPower2 - arrayLength;
        uint_t elemsPerThreadBlock = elemsPerThreadBlock = threadsPadding * elemsPadding;
        dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(threadsPadding, 1, 1);

        // Depending on sort order different value is used for padding.
        if (sortOrder == ORDER_ASC)
        {
            addPaddingKernel<MAX_VAL, elemsPadding><<<dimGrid, dimBlock>>>(
                d_keys, d_keysBuffer, arrayLength, paddingLength
            );
        }
        else
        {
            addPaddingKernel<MIN_VAL, elemsPadding><<<dimGrid, dimBlock>>>(
                d_keys, d_keysBuffer, arrayLength, paddingLength
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
            sharedMemSize = 2 * elemsIntervalsKo * threadBlockSize * sizeof(*intervals);
        }
        else
        {
            threadBlockSize = min((intervalsLen - 1) / elemsIntervalsKv + 1, threadsIntervalsKv);
            numThreadBlocks = (intervalsLen - 1) / (elemsIntervalsKv * threadBlockSize) + 1;
            // "2 *" because of BUFFER MEMORY for intervals
            sharedMemSize = 2 * elemsIntervalsKv * threadBlockSize * sizeof(*intervals);
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
    Runs kernel, whic performs bitonic merge from provided intervals.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitoicMergeIntervalsKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, interval_t *intervals,
        uint_t arrayLength, uint_t phasesBitonicMerge, uint_t phase
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

    template <order_t sortOrder, bool sortingKeyOnly>
    void bitonicSortAdaptiveParallel(
        data_t *h_keys, data_t *h_values, data_t *d_keys, data_t *d_values, data_t *d_keysBuffer,
        data_t *d_valuesBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, uint_t arrayLength
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

        runAddPaddingKernel<sortOrder>(d_keys, d_keysBuffer, arrayLength);
        runBitoicSortKernel<sortOrder, sortingKeyOnly>(d_keys, d_values, arrayLength);

        for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
        {
            uint_t stepStart = phase;
            uint_t stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
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

            // Global merge with intervals
            runBitoicMergeKernel<sortOrder, sortingKeyOnly>(
                d_keys, d_values, d_keysBuffer, d_valuesBuffer, d_intervals, arrayLength, phasesBitonicMerge, phase
            );

            // Exchanges keys
            data_t *tempTable = d_keys;
            d_keys = d_keysBuffer;
            d_keysBuffer = tempTable;

            // Exchanges values
            tempTable = d_values;
            d_values = d_valuesBuffer;
            d_valuesBuffer = tempTable;
        }
    }

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
Class for parallel adaptive bitonic sort.

uint_t threadsGenIntervalsKo, uint_t elemsGenIntervalsKo,
uint_t threadsGenIntervalsKv, uint_t elemsGenintervalsKv
*/
class BitonicSortAdaptiveParallel : public BitonicSortAdaptiveParallelBase<
    THREADS_BITONIC_SORT_KO, ELEMS_THREAD_BITONIC_SORT_KO,
    THREADS_BITONIC_SORT_KV, ELEMS_THREAD_BITONIC_SORT_KV,
    0, 0,  // Global bitonic merge is not needed
    0, 0,  // Global bitonic merge is not needed
    THREADS_LOCAL_MERGE_KO, ELEMS_THREAD_LOCAL_MERGE_KO,
    THREADS_LOCAL_MERGE_KV, ELEMS_THREAD_LOCAL_MERGE_KV,
    THREADS_PADDING, ELEMS_THREAD_PADDING,
    THREADS_INIT_INTERVALS_KO, ELEMS_INIT_INTERVALS_KO,
    THREADS_INIT_INTERVALS_KV, ELEMS_INIT_INTERVALS_KV,
    THREADS_GEN_INTERVALS_KO, ELEMS_GEN_INTERVALS_KO,
    THREADS_GEN_INTERVALS, ELEMS_GEN_INTERVALS_KV
>
{};

#endif
