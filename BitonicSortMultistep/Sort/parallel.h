#ifndef BITONIC_SORT_MULTISTEP_PARALLEL_H
#define BITONIC_SORT_MULTISTEP_PARALLEL_H

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../Utils/data_types_common.h"
#include "../../BitonicSort/Sort/parallel.h"
#include "../../Utils/host.h"
#include "../../Utils/cuda.h"
#include "../constants.h"

#define __CUDA_INTERNAL_COMPILATION__
#include "../Kernels/key_only.h"
#include "../Kernels/key_value.h"
#undef __CUDA_INTERNAL_COMPILATION__


/*
Base class for parallel multistep bitonic sort.
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
    uint_t threadsMultistepKo, uint_t maxMultistepKo,
    uint_t threadsMultistepKv, uint_t maxMultistepKv
>
class BitonicSortMultistepParallelBase : public BitonicSortParallelBase<
    threadsBitonicSortKo, elemsBitonicSortKo, threadsBitonicSortKv, elemsBitonicSortKv,
    threadsGlobalMergeKo, elemsGlobalMergeKo, threadsGlobalMergeKv, elemsGlobalMergeKv,
    threadsLocalMergeKo, elemsLocalMergeKo, threadsLocalMergeKv, elemsLocalMergeKv
>
{
protected:
    std::string _sortName = "Bitonic sort multistep parallel";

    /*
    Runs bitonic multistep merge kernel, which uses registers. Multistep means, that every thread reads
    multiple elements and sorts them according to bitonic sort exchanges for N steps ahead.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void runMultiStepKernel(
        data_t *d_keys, data_t *d_values, uint_t arrayLength, uint_t phase, uint_t step, uint_t degree
    )
    {
        // Breaks table len into its power of 2 length and the remainder.
        uint_t power2arrayLen = previousPowerOf2(arrayLength);
        uint_t residueArrayLen = arrayLength % power2arrayLen;

        uint_t partitionSize = (power2arrayLen - 1) / (1 << degree) + 1;
        // For remainder the size of partition has to be calculated explicitly, because it depends on
        // remainder size, step and degree
        if (residueArrayLen > 0)
        {
            // The size of one sub-block which is sorted with same group of comparissons.
            uint_t subBlockSize = 1 << step;
            // Rounds the residue size to the next power of sub-block size
            uint_t power2residueArrayLen = roundUp(residueArrayLen, subBlockSize);
            partitionSize += min(residueArrayLen, (power2residueArrayLen - 1) / (1 << degree) + 1);
        }

        uint_t threadBlockSize;
        if (sortingKeyOnly)
        {
            threadBlockSize = min(partitionSize, threadsMultistepKo);
        }
        else
        {
            threadBlockSize = min(partitionSize, threadsMultistepKv);
        }

        dim3 dimGrid((partitionSize - 1) / threadBlockSize + 1, 1, 1);
        dim3 dimBlock(threadBlockSize, 1, 1);

        if (sortingKeyOnly)
        {
            if (degree == 1)
            {
                multiStep1Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
            else if (degree == 2)
            {
                multiStep2Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
            else if (degree == 3)
            {
                multiStep3Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
            else if (degree == 4)
            {
                multiStep4Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
            else if (degree == 5)
            {
                multiStep5Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
            else if (degree == 6)
            {
                multiStep6Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, arrayLength, step);
            }
        }
        else
        {
            if (degree == 1)
            {
                multiStep1Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, d_values, arrayLength, step);
            }
            else if (degree == 2)
            {
                multiStep2Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, d_values, arrayLength, step);
            }
            else if (degree == 3)
            {
                multiStep3Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, d_values, arrayLength, step);
            }
            else if (degree == 4)
            {
                multiStep4Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, d_values, arrayLength, step);
            }
            else if (degree == 5)
            {
                multiStep5Kernel<sortOrder><<<dimGrid, dimBlock>>>(d_keys, d_values, arrayLength, step);
            }
        }
    }

    /*
    Sorts data with NORMALIZED MULTISTEP BITONIC SORT.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void bitonicSortMultistepParallel(data_t *d_keys, data_t *d_values, uint_t arrayLength)
    {
        uint_t arrayLengthPower2 = nextPowerOf2(arrayLength);
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
        uint_t phasesBitonicSort = log2((double)min(arrayLengthPower2, elemsPerBlockBitonicSort));
        uint_t phasesMergeLocal = log2((double)min(arrayLengthPower2, elemsPerBlockMergeLocal));
        uint_t phasesAll = log2((double)arrayLengthPower2);

        runBitoicSortKernel<sortOrder, sortingKeyOnly>(
            d_keys, d_values, arrayLength
        );

        // Bitonic merge
        for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
        {
            uint_t step = phase;

            if (step > phasesMergeLocal)
            {
                // Global NORMALIZED bitonic merge for first step of phase, where different pattern of exchanges
                // is used compared to other steps
                runBitonicMergeGlobalKernel<sortOrder, sortingKeyOnly>(
                    d_keys, d_values, arrayLength, phase, step
                );

                step--;
                uint_t maxMultistep = sortingKeyOnly ? maxMultistepKo : maxMultistepKv;

                // Multisteps
                for (uint_t degree = min(maxMultistep, step - phasesMergeLocal); degree > 0; degree--)
                {
                    for (; step >= phasesMergeLocal + degree; step -= degree)
                    {
                        runMultiStepKernel<sortOrder, sortingKeyOnly>(
                            d_keys, d_values, arrayLength, phase, step, degree
                        );
                    }
                }
            }

            runBitoicMergeLocalKernel<sortOrder, sortingKeyOnly>(
                d_keys, d_values, arrayLength, phase, step
            );
        }
    }

    /*
    Wrapper for multistep bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyOnly()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortMultistepParallel<ORDER_ASC, true>(_d_keys, NULL, _arrayLength);
        }
        else
        {
            bitonicSortMultistepParallel<ORDER_DESC, true>(_d_keys, NULL, _arrayLength);
        }
    }

    /*
    Wrapper for multistep bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            bitonicSortMultistepParallel<ORDER_ASC, false>(_d_keys, _d_values, _arrayLength);
        }
        else
        {
            bitonicSortMultistepParallel<ORDER_DESC, false>(_d_keys, _d_values, _arrayLength);
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
class BitonicSortMultistepParallel : public BitonicSortMultistepParallelBase<
    THREADS_BITONIC_SORT_KO, ELEMS_BITONIC_SORT_KO,
    THREADS_BITONIC_SORT_KV, ELEMS_BITONIC_SORT_KV,
    THREADS_GLOBAL_MERGE_KO, ELEMS_GLOBAL_MERGE_KO,
    THREADS_GLOBAL_MERGE_KV, ELEMS_GLOBAL_MERGE_KV,
    THREADS_LOCAL_MERGE_KO, ELEMS_LOCAL_MERGE_KO,
    THREADS_LOCAL_MERGE_KV, ELEMS_LOCAL_MERGE_KV,
    THREADS_MULTISTEP_MERGE_KO, MAX_MULTI_STEP_KO,
    THREADS_MULTISTEP_MERGE_KV, MAX_MULTI_STEP_KV
>
{};

#endif
