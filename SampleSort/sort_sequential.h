#ifndef SAMPLE_SORT_SEQUENTIAL_H
#define SAMPLE_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <functional>
#include <chrono>

#include "../Utils/data_types_common.h"
#include "../Utils/sort_correct.h"
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

    /*
    From provided array collects "numSamples" samples and sorts them.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void collectSamples(data_t *d_keys, data_t *h_samples, uint_t arrayLength)
    {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        auto generator = std::bind(std::uniform_int_distribution<uint_t>(0, arrayLength - 1), mt19937(seed));
        uint_t numSamples = sortingKeyOnly ? numSamplesKo : numSamplesKv;

        // Collects "numSamples" samples
        for (uint_t i = 0; i < numSamples; i++)
        {
            h_samples[i] = d_keys[generator()];
        }

        // Samples are sorted with in-place sort
        stdVectorSort<data_t>(h_samples, numSamples, sortOrder);
    }

    /*
    Performs inclusive binary search and returns index where element should be located.
    */
    template <order_t sortOrder>
    int binarySearchInclusive(data_t* h_keys, data_t target, uint_t arrayLength)
    {
        int_t indexStart = 0;
        int_t indexEnd = arrayLength - 1;

        while (indexStart <= indexEnd)
        {
            int index = (indexStart + indexEnd) / 2;

            if (sortOrder == ORDER_ASC ? (target <= h_keys[index]) : (target >= h_keys[index]))
            {
                indexEnd = index - 1;
            }
            else
            {
                indexStart = index + 1;
            }
        }

        return indexStart;
    }

    /*
    Performs EXCLUSIVE scan in-place on provided array.
    */
    void exclusiveScan(uint_t *dataArray, uint_t arrayLength)
    {
        uint_t prevElem = dataArray[0];
        dataArray[0] = 0;

        for (uint_t i = 1; i < arrayLength; i++)
        {
            uint_t currElem = dataArray[i];
            dataArray[i] = dataArray[i - 1] + prevElem;
            prevElem = currElem;
        }
    }

    ///*
    //Sorts array with sample sort and outputs sorted data to result array.
    //*/
    //template <order_t sortOrder>
    //void sampleSort(
    //    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, data_t *resultKeys,
    //    data_t *resultValues, data_t *samples, uint_t *elementBuckets, uint_t tableLen
    //)
    //{
    //    // When array is small enough, it is sorted with small sort (in our case merge sort).
    //    // Merge sort was chosen because it is stable sort and it keeps sorted array stable.
    //    if (tableLen <= SMALL_SORT_THRESHOLD)
    //    {
    //        mergeSort<sortOrder>(dataKeys, dataValues, bufferKeys, bufferValues, resultKeys, resultValues, tableLen);
    //        return;
    //    }
    //
    //    collectSamples<sortOrder>(dataKeys, samples, tableLen);
    //
    //    // Holds bucket sizes and bucket offsets after exclusive scan is performed on bucket sizes.
    //    // A new array is needed for every level of recursion.
    //    uint_t bucketSizes[NUM_SPLITTERS_SEQUENTIAL + 1];
    //    // For clarity purposes another pointer is used
    //    data_t *splitters = samples;
    //
    //    // From "NUM_SAMPLES_SEQUENTIAL" samples collects "NUM_SPLITTERS_SEQUENTIAL" splitters
    //    for (uint_t i = 0; i < NUM_SPLITTERS_SEQUENTIAL; i++)
    //    {
    //        splitters[i] = samples[i * OVERSAMPLING_FACTOR + (OVERSAMPLING_FACTOR / 2)];
    //        bucketSizes[i] = 0;
    //    }
    //    // For "NUM_SPLITTERS_SEQUENTIAL" splitters "NUM_SPLITTERS_SEQUENTIAL + 1" buckets are created
    //    bucketSizes[NUM_SPLITTERS_SEQUENTIAL] = 0;
    //
    //    // For all elements in data table searches, which bucket they belong to and counts the elements in buckets
    //    for (uint_t i = 0; i < tableLen; i++)
    //    {
    //        uint_t bucket = binarySearchInclusive<sortOrder>(splitters, dataKeys[i], NUM_SPLITTERS_SEQUENTIAL);
    //        bucketSizes[bucket]++;
    //        elementBuckets[i] = bucket;
    //    }
    //
    //    // Performs an EXCLUSIVE scan over array of bucket sizes in order to get bucket offsets
    //    exclusiveScan(bucketSizes, NUM_SPLITTERS_SEQUENTIAL + 1);
    //    // For clarity purposes another pointer is used
    //    uint_t *bucketOffsets = bucketSizes;
    //
    //    // Goes through all elements again and stores them in their corresponding buckets
    //    for (uint_t i = 0; i < tableLen; i++)
    //    {
    //        uint_t *bucketOffset = &bucketOffsets[elementBuckets[i]];
    //        bufferKeys[*bucketOffset] = dataKeys[i];
    //        bufferValues[*bucketOffset] = dataValues[i];
    //
    //        (*bucketOffset)++;
    //    }
    //
    //    // Recursively sorts buckets
    //    for (uint_t i = 0; i <= NUM_SPLITTERS_SEQUENTIAL; i++)
    //    {
    //        uint_t prevBucketOffset = i > 0 ? bucketOffsets[i - 1] : 0;
    //        uint_t bucketSize = bucketOffsets[i] - prevBucketOffset;
    //
    //        if (bucketSize > 0)
    //        {
    //            // Primary and buffer array are exchanged
    //            sampleSort<sortOrder>(
    //                bufferKeys + prevBucketOffset, bufferValues + prevBucketOffset, dataKeys + prevBucketOffset,
    //                dataValues + prevBucketOffset, resultKeys + prevBucketOffset, resultValues + prevBucketOffset,
    //                samples, elementBuckets, bucketSize
    //            );
    //        }
    //    }
    //}

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
