#ifndef SAMPLE_SORT_SEQUENTIAL_H
#define SAMPLE_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../MergeSort/sort_sequential.h"
#include "constants.h"


/*
Parent class for sequential sample sort. Not to be used directly - it's inherited by bottom class, which performs
partial template specialization.

Template params:
_Ko - Key-only
_Kv - Key-value
*/
template <
    uint_t numSplittersKo, uint_t numSplittersKv,
    uint_t numSamplesKo, uint_t numSamplesKv,
    uint_t oversamplingFactorKo, uint_t oversamplingFactorKv,
    uint_t smallSortThresholdKo, uint_t smallSortThresholdKv
>
class SampleSortSequentialParent : public MergeSortSequential
{
protected:
    std::string _sortName = "Sample sort sequential";

    // Holds samples and after samples are sorted holds splitters in sequential sample sort
    data_t *_h_samples;
    // For every element in input holds bucket index to which it belogns (needed for sequential sample sort)
    uint_t *_h_elementBuckets;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        MergeSortSequential::memoryAllocate(h_keys, h_values, arrayLength);

        uint_t maxNumSamples = max(numSamplesKo, numSamplesKv);
        uint_t maxOversamplingFactor = max(oversamplingFactorKo, oversamplingFactorKv);

        // Holds samples and splitters in sequential sample sort (needed for sequential sample sort)
        _h_samples = (data_t*)malloc(maxNumSamples * maxOversamplingFactor * sizeof(*_h_samples));
        checkMallocError(_h_samples);
        // For each element in array holds, to which bucket it belongs (needed for sequential sample sort)
        _h_elementBuckets = (uint_t*)malloc(arrayLength * sizeof(*_h_elementBuckets));
        checkMallocError(_h_elementBuckets);
    }

    ///*
    //Wrapper for bitonic sort method.
    //The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    //*/
    //void sortKeyOnly()
    //{
    //    if (_sortOrder == ORDER_ASC)
    //    {
    //        mergeSortSequential<ORDER_ASC, true>(_h_keys, NULL, _h_keysBuffer, NULL, _arrayLength);
    //    }
    //    else
    //    {
    //        mergeSortSequential<ORDER_DESC, true>(_h_keys, NULL, _h_keysBuffer, NULL, _arrayLength);
    //    }
    //}

    ///*
    //Wrapper for bitonic sort method.
    //The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    //*/
    //void sortKeyValue()
    //{
    //    if (_sortOrder == ORDER_ASC)
    //    {
    //        mergeSortSequential<ORDER_ASC, false>(_h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _arrayLength);
    //    }
    //    else
    //    {
    //        mergeSortSequential<ORDER_DESC, false>(_h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _arrayLength);
    //    }
    //}

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

        MergeSortSequential::memoryDestroy();

        free(_h_samples);
        free(_h_valuesBuffer);
    }
};

/*
Base class for sequential sample sort.
*/
template <
    uint_t numSplittersKo, uint_t numSplittersKv,
    uint_t oversamplingFactorKo, uint_t oversamplingFactorKv,
    uint_t smallSortThresholdKo, uint_t smallSortThresholdKv
>
class SampleSortSequentialBase : public SampleSortSequentialParent<
    numSplittersKo, numSplittersKv,
    numSplittersKo * oversamplingFactorKo, numSplittersKv * oversamplingFactorKv,
    oversamplingFactorKo, oversamplingFactorKv,
    smallSortThresholdKo, smallSortThresholdKv
>
{};

/*
Class for sequential sample sort.
*/
class SampleSortSequential : public SampleSortSequentialBase<
    NUM_SPLITTERS_SEQUENTIAL_KO, NUM_SPLITTERS_SEQUENTIAL_KV,
    OVERSAMPLING_FACTOR_KO, OVERSAMPLING_FACTOR_KV,
    SMALL_SORT_THRESHOLD_KO, SMALL_SORT_THRESHOLD_KV
>
{};

#endif
