#ifndef BITONIC_SORT_PARALLEL_H
#define BITONIC_SORT_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_interface.h"
#include "../../Utils/host.h"
#include "../constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "../Kernels/key_only.h"
#include "../Kernels/key_value.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Base class for parallel bitonic sort.
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
    uint_t threadsLocalMergeKv, uint_t elemsLocalMergeKv
>
class BitonicSortParallelBase : public SortParallel
{
protected:
    std::string _sortName = "Bitonic sort parallel";

    /*
    Sorts sub-blocks of input data with bitonic sort.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitoicSortKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength)
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
            bitonicSortKernel
                <threadsBitonicSortKo, elemsBitonicSortKo, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, arrayLength
            );
        }
        else
        {
            bitonicSortKernel
                <threadsBitonicSortKv, elemsBitonicSortKv, sortOrder><<<dimGrid, dimBlock, sharedMemSize>>>(
                d_keys, d_values, arrayLength
            );
        }
    }

    /*
    Merges array, if data blocks are larger than shared memory size. It executes only one STEP of one PHASE per
    kernel launch.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitonicMergeGlobalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step)
    {
        uint_t elemsPerThreadBlock;
        if (sortingKeyOnly)
        {
            elemsPerThreadBlock = threadsGlobalMergeKo * elemsGlobalMergeKo;
        }
        else
        {
            elemsPerThreadBlock = threadsGlobalMergeKv * elemsGlobalMergeKv;
        }

        dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsGlobalMergeKo : threadsGlobalMergeKv, 1, 1);

        bool isFirstStepOfPhase = phase == step;

        if (sortingKeyOnly)
        {
            if (isFirstStepOfPhase)
            {
                bitonicMergeGlobalKernel
                    <threadsGlobalMergeKo, elemsGlobalMergeKo, sortOrder, true><<<dimGrid, dimBlock>>>(
                    d_keys, arrayLength, step
                );
            }
            else
            {
                bitonicMergeGlobalKernel
                    <threadsGlobalMergeKo, elemsGlobalMergeKo, sortOrder, false><<<dimGrid, dimBlock>>>(
                    d_keys, arrayLength, step
                );
            }
        }
        else
        {
            if (isFirstStepOfPhase)
            {
                bitonicMergeGlobalKernel
                    <threadsGlobalMergeKv, elemsGlobalMergeKv, sortOrder, true><<<dimGrid, dimBlock>>>(
                    d_keys, d_values, arrayLength, step
                );
            }
            else
            {
                bitonicMergeGlobalKernel
                    <threadsGlobalMergeKv, elemsGlobalMergeKv, sortOrder, false><<<dimGrid, dimBlock>>>(
                    d_keys, d_values, arrayLength, step
                );
            }
        }
    }

    /*
    Merges array when stride is lower than shared memory size. It executes all remaining STEPS of current PHASE.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runBitoicMergeLocalKernel(data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step)
    {
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

        dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
        dim3 dimBlock(sortingKeyOnly ? threadsLocalMergeKo : threadsLocalMergeKv, 1, 1);

        bool isFirstStepOfPhase = phase == step;

        if (sortingKeyOnly)
        {
            if (isFirstStepOfPhase) {
                bitonicMergeLocalKernel
                    <threadsLocalMergeKo, elemsLocalMergeKo, sortOrder, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_keys, arrayLength, step
                );
            }
            else
            {
                bitonicMergeLocalKernel
                    <threadsLocalMergeKo, elemsLocalMergeKo, sortOrder, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_keys, arrayLength, step
                );
            }
        }
        else
        {
            if (isFirstStepOfPhase) {
                bitonicMergeLocalKernel
                    <threadsLocalMergeKv, elemsLocalMergeKv, sortOrder, true><<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_keys, d_values, arrayLength, step
                );
            }
            else
            {
                bitonicMergeLocalKernel
                    <threadsLocalMergeKv, elemsLocalMergeKv, sortOrder, false><<<dimGrid, dimBlock, sharedMemSize>>>(
                    d_keys, d_values, arrayLength, step
                );
            }
        }
    }

    /*
    Sorts data with parallel NORMALIZED BITONIC SORT.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void bitonicSortParallel(data_t *d_keys, data_t *d_values, uint_t arrayLength)
    {
        uint_t arrayLenPower2 = nextPowerOf2(arrayLength);
        uint_t elemsPerBlockBitonicSort, elemsPerBlockMergeLocal;

        if (sortingKeyOnly)
        {
            elemsPerBlockBitonicSort = threadsBitonicSortKo * elemsBitonicSortKo;
            elemsPerBlockMergeLocal = threadsLocalMergeKo * elemsLocalMergeKo;
        }
        else
        {
            elemsPerBlockBitonicSort = threadsBitonicSortKv * elemsBitonicSortKv;
            elemsPerBlockMergeLocal = threadsLocalMergeKv * elemsLocalMergeKv;
        }

        // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
        uint_t phasesBitonicSort = log2((double)min(arrayLenPower2, elemsPerBlockBitonicSort));
        uint_t phasesMergeLocal = log2((double)min(arrayLenPower2, elemsPerBlockMergeLocal));
        uint_t phasesAll = log2((double)arrayLenPower2);

        // Sorts blocks of input data with bitonic sort
        runBitoicSortKernel<sortOrder, sortingKeyOnly>(
            d_keys, d_values, arrayLength
        );

        // Bitonic merge
        for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
        {
            uint_t step = phase;
            while (step > phasesMergeLocal)
            {
                runBitonicMergeGlobalKernel<sortOrder, sortingKeyOnly>(
                    d_keys, d_values, arrayLength, phase, step
                );
                step--;
            }

            runBitoicMergeLocalKernel<sortOrder, sortingKeyOnly>(d_keys, d_values, arrayLength, phase, step);
        }
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortParallel<ORDER_ASC, true>(_d_keys, NULL, _arrayLength);
        }
        else
        {
            bitonicSortParallel<ORDER_DESC, true>(_d_keys, NULL, _arrayLength);
        }
    }

    /*
    wrapper for bitonic sort method.
    the code runs faster if arguments are passed to method. if members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortParallel<ORDER_ASC, false>(_d_keys, _d_values, _arrayLength);
        }
        else
        {
            bitonicSortParallel<ORDER_DESC, false>(_d_keys, _d_values, _arrayLength);
        }
    }

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};


/*
Class for parallel bitonic sort.
*/
class BitonicSortParallel : public BitonicSortParallelBase<
    THREADS_BITONIC_SORT_KO, ELEMS_BITONIC_SORT_KO,
    THREADS_BITONIC_SORT_KV, ELEMS_BITONIC_SORT_KV,
    THREADS_GLOBAL_MERGE_KO, ELEMS_GLOBAL_MERGE_KO,
    THREADS_GLOBAL_MERGE_KV, ELEMS_GLOBAL_MERGE_KV,
    THREADS_LOCAL_MERGE_KO, ELEMS_LOCAL_MERGE_KO,
    THREADS_LOCAL_MERGE_KV, ELEMS_LOCAL_MERGE_KV
>
{};

#endif
