#ifndef SAMPLE_SORT_SEQUENTIAL_H
#define SAMPLE_SORT_SEQUENTIAL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <functional>
#include <chrono>

#include "../../Utils/data_types_common.h"
#include "../../Utils/sort_correct.h"
#include "../../Utils/host.h"
#include "../../MergeSort/Sort/sequential.h"
#include "../constants.h"


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

    // Arrays where sorted sequence is saved. It's needed because subsequences are moved from primary and buffer
    // memory. This way some of the sorted array would end up in primary array and other part in buffer.
    data_t *_h_keysSorted = NULL, *_h_valuesSorted = NULL;
    // Holds samples and after samples are sorted holds splitters in sequential sample sort
    data_t *_h_samples;
    // For every element in input holds bucket index to which it belongs (needed for sequential sample sort)
    uint_t *_h_elementBuckets;

    /*
    Method for allocating memory needed both for key only and key-value sort.
    */
    virtual void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        MergeSortSequential::memoryAllocate(h_keys, h_values, arrayLength);

        uint_t maxNumSamples = max(numSamplesKo, numSamplesKv);

        _h_keysSorted = (data_t*)malloc(arrayLength * sizeof(*_h_keysSorted));
        checkMallocError(_h_keysSorted);
        _h_valuesSorted = (data_t*)malloc(arrayLength * sizeof(*_h_valuesSorted));
        checkMallocError(_h_valuesSorted);

        // Holds samples and splitters in sequential sample sort (needed for sequential sample sort)
        _h_samples = (data_t*)malloc(maxNumSamples * sizeof(*_h_samples));
        checkMallocError(_h_samples);
        // For each element in array holds, to which bucket it belongs (needed for sequential sample sort)
        _h_elementBuckets = (uint_t*)malloc(arrayLength * sizeof(*_h_elementBuckets));
        checkMallocError(_h_elementBuckets);
    }

    /*
    Sorted sequence is located in sorted array.
    */
    virtual void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength)
    {
        std::copy(_h_keysSorted, _h_keysSorted + arrayLength, h_keys);
        if (h_values != NULL)
        {
            std::copy(_h_valuesSorted, _h_valuesSorted + arrayLength, h_values);
        }
    }

    /*
    Returns the output array. Implemented because sample sort class (which is derived from this class) requires
    different function to determine output array than this class.
    */
    virtual data_t* getOutputMergeArray(data_t *arrayBuffer, data_t *arraySorted, bool isLastMergePhase)
    {
        return isLastMergePhase ? arraySorted : arrayBuffer;
    }

    /*
    From provided array collects "numSamples" samples and sorts them.
    */
    template <order_t sortOrder, bool sortingKeyOnly>
    void collectSamples(data_t *d_keys, data_t *h_samples, uint_t arrayLength)
    {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        auto generator = std::bind(std::uniform_int_distribution<uint_t>(0, arrayLength - 1), std::mt19937(seed));
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

    /*
    Sorts array with sample sort and outputs sorted data to result array.
    */
    template <
        order_t sortOrder, uint_t sortingKeyOnly, uint_t numSplitters, uint_t oversamplingFactor,
        uint_t smallSortThreashold
    >
    void sampleSortSequential(
        data_t *h_keys, data_t *h_values, data_t *h_keysBuffer, data_t *h_valuesBuffer, data_t *h_keysSorted,
        data_t *h_valuesSorted, data_t *h_samples, uint_t *h_elementBuckets, uint_t arrayLength
    )
    {
        // When array is small enough, it is sorted with small sort (in our case merge sort).
        // Merge sort was chosen because it is stable sort and it keeps sorted array stable.
        if (arrayLength <= smallSortThreashold)
        {
            mergeSortSequential<sortOrder, sortingKeyOnly>(
                h_keys, h_values, h_keysBuffer, h_valuesBuffer, h_keysSorted, h_valuesSorted, arrayLength
            );
            return;
        }

        collectSamples<sortOrder, sortingKeyOnly>(h_keys, h_samples, arrayLength);

        // Holds bucket sizes and bucket offsets after exclusive scan is performed on bucket sizes.
        // A new array is needed for every level of recursion.
        uint_t bucketSizes[numSplitters + 1];
        // For clarity purposes another pointer is used
        data_t *splitters = h_samples;

        // From "NUM_SAMPLES_SEQUENTIAL" samples collects "numSplitters" splitters
        for (uint_t i = 0; i < numSplitters; i++)
        {
            splitters[i] = h_samples[i * oversamplingFactor + (oversamplingFactor / 2)];
            bucketSizes[i] = 0;
        }
        // For "numSplitters" splitters "numSplitters + 1" buckets are created
        bucketSizes[numSplitters] = 0;

        // For all elements in data table searches, which bucket they belong to and counts the elements in buckets
        for (uint_t i = 0; i < arrayLength; i++)
        {
            uint_t bucket = binarySearchInclusive<sortOrder>(splitters, h_keys[i], numSplitters);
            bucketSizes[bucket]++;
            h_elementBuckets[i] = bucket;
        }

        // Performs an EXCLUSIVE scan over array of bucket sizes in order to get bucket offsets
        exclusiveScan(bucketSizes, numSplitters + 1);
        // For clarity purposes another pointer is used
        uint_t *bucketOffsets = bucketSizes;

        // Goes through all elements again and stores them in their corresponding buckets
        for (uint_t i = 0; i < arrayLength; i++)
        {
            uint_t *bucketOffset = &bucketOffsets[h_elementBuckets[i]];
            h_keysBuffer[*bucketOffset] = h_keys[i];

            if (!sortingKeyOnly)
            {
                h_valuesBuffer[*bucketOffset] = h_values[i];
            }

            (*bucketOffset)++;
        }

        // Recursively sorts buckets
        for (uint_t i = 0; i <= numSplitters; i++)
        {
            uint_t prevBucketOffset = i > 0 ? bucketOffsets[i - 1] : 0;
            uint_t bucketSize = bucketOffsets[i] - prevBucketOffset;

            // Without this condition recursion would never end, if distribution was ZERO (all elements are same).
            if (bucketSize == arrayLength)
            {
                mergeSortSequential<sortOrder, sortingKeyOnly>(
                    h_keysBuffer, h_valuesBuffer, h_keys, h_values, h_keysSorted, h_valuesSorted, arrayLength
                );
                return;
            }

            if (bucketSize > 0)
            {
                // Primary and buffer arrays are exchanged
                if (sortingKeyOnly)
                {
                    sampleSortSequential
                        <sortOrder, sortingKeyOnly, numSplitters, oversamplingFactor, smallSortThreashold>(
                        h_keysBuffer + prevBucketOffset, NULL, h_keys + prevBucketOffset, NULL,
                        h_keysSorted + prevBucketOffset, NULL, h_samples, h_elementBuckets, bucketSize
                    );
                }
                else
                {
                    sampleSortSequential
                        <sortOrder, sortingKeyOnly, numSplitters, oversamplingFactor, smallSortThreashold>(
                        h_keysBuffer + prevBucketOffset, h_valuesBuffer + prevBucketOffset, h_keys + prevBucketOffset,
                        h_values + prevBucketOffset, h_keysSorted + prevBucketOffset,
                        h_valuesSorted + prevBucketOffset, h_samples, h_elementBuckets, bucketSize
                    );
                }
            }
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
            sampleSortSequential<ORDER_ASC, true, numSplittersKo, oversamplingFactorKo, smallSortThresholdKo>(
                _h_keys, NULL, _h_keysBuffer, NULL, _h_keysSorted, NULL, _h_samples, _h_elementBuckets,
                _arrayLength
            );
        }
        else
        {
            sampleSortSequential<ORDER_DESC, true, numSplittersKo, oversamplingFactorKo, smallSortThresholdKo>(
                _h_keys, NULL, _h_keysBuffer, NULL, _h_keysSorted, NULL, _h_samples, _h_elementBuckets,
                _arrayLength
            );
        }
    }

    /*
    Wrapper for bitonic sort method.
    The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
    */
    void sortKeyValue()
    {
        if (_sortOrder == ORDER_ASC)
        {
            sampleSortSequential<ORDER_ASC, false, numSplittersKv, oversamplingFactorKv, smallSortThresholdKv>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _h_keysSorted, _h_valuesSorted, _h_samples,
                _h_elementBuckets, _arrayLength
            );
        }
        else
        {
            sampleSortSequential<ORDER_DESC, false, numSplittersKv, oversamplingFactorKv, smallSortThresholdKv>(
                _h_keys, _h_values, _h_keysBuffer, _h_valuesBuffer, _h_keysSorted, _h_valuesSorted, _h_samples,
                _h_elementBuckets, _arrayLength
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

        MergeSortSequential::memoryDestroy();

        free(_h_keysSorted);
        free(_h_valuesSorted);
        free(_h_samples);
        free(_h_elementBuckets);
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
