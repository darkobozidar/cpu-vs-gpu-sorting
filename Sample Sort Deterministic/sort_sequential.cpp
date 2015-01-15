#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iostream>
#include <functional>
#include <chrono>
#include <stdint.h>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "constants.h"

using namespace std;


/* ---------------- MERGE SORT ---------------- */

/*
From provided table offset, size of table block and length of entire table returns end index of the block.
*/
uint_t getEndIndex(uint_t offset, uint_t subBlockSize, uint_t tableLen)
{
    uint_t endIndex = offset + subBlockSize;
    return endIndex <= tableLen ? endIndex : tableLen;
}

/*
Sorts data sequentially with merge sort. Sorted array is outputted to result array.
Stable sort (merge sort) is used in order to keep sample sort stable.
*/
template <order_t sortOrder>
void mergeSort(data_t *dataTable, data_t *dataBuffer, data_t *dataResult, uint_t tableLen)
{
    if (tableLen == 1)
    {
        dataResult[0] = dataTable[0];
        return;
    }

    uint_t tableLenPower2 = nextPowerOf2(tableLen);

    // Log(tableLen) phases of merge sort
    for (uint_t sortedBlockSize = 2; sortedBlockSize <= tableLenPower2; sortedBlockSize *= 2)
    {
        // Number of merged blocks that will be created in this iteration
        uint_t numBlocks = (tableLen - 1) / sortedBlockSize + 1;
        // Number of sub-blocks being merged
        uint_t subBlockSize = sortedBlockSize / 2;
        // If it is last phase of merge sort, data is coppied to result array
        data_t *outputTable = numBlocks == 1 ? dataResult : dataBuffer;

        // Merge of all blocks
        for (uint_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
        {
            // Odd (left) block being merged
            uint_t oddIndex = blockIndex * sortedBlockSize;
            uint_t oddEnd = getEndIndex(oddIndex, subBlockSize, tableLen);

            // If there is only odd block without even block, then only odd block is coppied into buffer
            if (oddEnd == tableLen)
            {
                std::copy(dataTable + oddIndex, dataTable + oddEnd, outputTable + oddIndex);
                continue;
            }

            // Even (right) block being merged
            uint_t evenIndex = oddIndex + subBlockSize;
            uint_t evenEnd = getEndIndex(evenIndex, subBlockSize, tableLen);
            uint_t mergeIndex = oddIndex;

            // Merge of odd and even block
            while (oddIndex < oddEnd && evenIndex < evenEnd)
            {
                data_t oddElement = dataTable[oddIndex];
                data_t evenElement = dataTable[evenIndex];

                if (sortOrder == ORDER_ASC ? oddElement <= evenElement : oddElement >= evenElement)
                {
                    outputTable[mergeIndex++] = oddElement;
                    oddIndex++;
                }
                else
                {
                    outputTable[mergeIndex++] = evenElement;
                    evenIndex++;
                }
            }

            // Block that wasn't merged entirely is coppied into buffer array
            if (oddIndex == oddEnd)
            {
                std::copy(dataTable + evenIndex, dataTable + evenEnd, outputTable + mergeIndex);
            }
            else
            {
                std::copy(dataTable + oddIndex, dataTable + oddEnd, outputTable + mergeIndex);
            }
        }

        data_t *temp = dataTable;
        dataTable = dataBuffer;
        dataBuffer = temp;
    }
}


/* ---------------- QUICK SORT ---------------- */

/*
Compare function for ASCENDING order needed for C++ qsort.
*/
int compareAsc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem1) - *((data_t*)elem2);
}

/*
 Compare function for DESCENDING order needed for C++ qsort.
 */
int compareDesc(const void* elem1, const void* elem2)
{
    return *((data_t*)elem2) - *((data_t*)elem1);
}

/*
Sorts an array with C quicksort implementation.
*/
template <order_t sortOrder>
void quickSort(data_t *dataTable, uint_t tableLen)
{
    if (sortOrder == ORDER_ASC)
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
    }
    else
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
    }
}


/* -------------- GENERAL UTILS --------------- */

/*
From provided data table collects "NUM_SAMPLES_SEQUENTIAL" samples and sorts them.
*/
template <order_t sortOrder>
void collectSamples(data_t *dataTable, data_t *samples, uint_t tableLen)
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    auto generator = std::bind(std::uniform_int_distribution<uint_t>(0, tableLen - 1), mt19937(seed));

    // Collects "NUM_SAMPLES_SEQUENTIAL" samples
    for (uint_t i = 0; i < NUM_SAMPLES_SEQUENTIAL; i++)
    {
        samples[i] = dataTable[generator()];
    }

    // Sorts samples with quicksort. Quicksort is used, because it is not necessary, than sort is stable.
    quickSort<sortOrder>(samples, NUM_SAMPLES_SEQUENTIAL);
}

/*
Performs inclusive binary search and returns index where element should be located.
*/
template <order_t sortOrder>
int binarySearchInclusive(data_t* dataTable, data_t target, uint_t tableLen)
{
    int_t indexStart = 0;
    int_t indexEnd = tableLen - 1;

    while (indexStart <= indexEnd)
    {
        int index = (indexStart + indexEnd) / 2;

        if (sortOrder == ORDER_ASC ? (target <= dataTable[index]) : (target >= dataTable[index]))
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
void exclusiveScan(uint_t *dataTable, uint_t tableLen)
{
    uint_t prevElem = dataTable[0];
    dataTable[0] = 0;

    for (uint_t i = 1; i < tableLen; i++)
    {
        uint_t currElem = dataTable[i];
        dataTable[i] = dataTable[i - 1] + prevElem;
        prevElem = currElem;
    }
}


/* ---------------- SAMPLE SORT --------------- */

/*
Sorts array with sample sort and outputs sorted data to result array.
*/
template <order_t sortOrder>
void sampleSort(
    data_t *dataTable, data_t *dataBuffer, data_t *dataResult, data_t *samples, uint_t *elementBuckets,
    uint_t tableLen
)
{
    // When array is small enough, it is sorted with small sort (in our case merge sort).
    // Merge sort was chosen because it is stable sort and it keeps sorted array stable.
    if (tableLen <= SMALL_SORT_THRESHOLD)
    {
        mergeSort<sortOrder>(dataTable, dataBuffer, dataResult, tableLen);
        return;
    }

    collectSamples<sortOrder>(dataTable, samples, tableLen);

    // Holds bucket sizes and bucket offsets after exclusive scan is performed on bucket sizes.
    // A new array is needed for every level of recursion.
    uint_t bucketSizes[NUM_SPLITTERS_SEQUENTIAL + 1];
    // For clarity purposes another pointer is used
    data_t *splitters = samples;

    // From "NUM_SAMPLES_SEQUENTIAL" samples collects "NUM_SPLITTERS_SEQUENTIAL" splitters
    for (uint_t i = 0; i < NUM_SPLITTERS_SEQUENTIAL; i++)
    {
        splitters[i] = samples[i * OVERSAMPLING_FACTOR + (OVERSAMPLING_FACTOR / 2)];
        bucketSizes[i] = 0;
    }
    // For "NUM_SPLITTERS_SEQUENTIAL" splitters "NUM_SPLITTERS_SEQUENTIAL + 1" buckets are created
    bucketSizes[NUM_SPLITTERS_SEQUENTIAL] = 0;

    // For all elements in data table searches, which bucket they belong to and counts the elements in buckets
    for (uint_t i = 0; i < tableLen; i++)
    {
        uint_t bucket = binarySearchInclusive<sortOrder>(splitters, dataTable[i], NUM_SPLITTERS_SEQUENTIAL);
        bucketSizes[bucket]++;
        elementBuckets[i] = bucket;
    }

    // Performs an EXCLUSIVE scan over array of bucket sizes in order to get bucket offsets
    exclusiveScan(bucketSizes, NUM_SPLITTERS_SEQUENTIAL + 1);
    // For clarity purposes another pointer is used
    uint_t *bucketOffsets = bucketSizes;

    // Goes through all elements again and stores them in their corresponding buckets
    for (uint_t i = 0; i < tableLen; i++)
    {
        dataBuffer[bucketOffsets[elementBuckets[i]]++] = dataTable[i];
    }

    // Recursively sorts buckets
    for (uint_t i = 0; i <= NUM_SPLITTERS_SEQUENTIAL; i++)
    {
        uint_t prevBucketOffset = i > 0 ? bucketOffsets[i - 1] : 0;
        uint_t bucketSize = bucketOffsets[i] - prevBucketOffset;

        if (bucketSize > 0)
        {
            // Primary and buffer array are exchanged
            sampleSort<sortOrder>(
                dataBuffer + prevBucketOffset, dataTable + prevBucketOffset, dataResult + prevBucketOffset, samples,
                elementBuckets, bucketSize
            );
        }
    }
}

/*
Sorts data sequentially with sample sort.
*/
double sortSequential(
    data_t *dataInput, data_t *dataBuffer, data_t *dataOutput, data_t *samples, uint_t *elementBuckets,
    uint_t tableLen, order_t sortOrder
)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    if (sortOrder == ORDER_ASC)
    {
        sampleSort<ORDER_ASC>(dataInput, dataBuffer, dataOutput, samples, elementBuckets, tableLen);
    }
    else
    {
        sampleSort<ORDER_DESC>(dataInput, dataBuffer, dataOutput, samples, elementBuckets, tableLen);
    }

    return endStopwatch(timer);
}
