#ifndef KERNELS_COMMON_MERGE_SORT_H
#define KERNELS_COMMON_MERGE_SORT_H

#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "../../Utils/data_types_common.h"
#include "../../Utils/kernels_utils.h"


/*
Generates array of ranks/boundaries of sub-block, which will be merged.
*/
template <uint_t subBlockSize, order_t sortOrder>
__global__ void generateRanksKernel(data_t *keys, uint_t *ranksEven, uint_t *ranksOdd, uint_t sortedBlockSize)
{
    uint_t subBlocksPerSortedBlock = sortedBlockSize / subBlockSize;
    uint_t subBlocksPerMergedBlock = 2 * subBlocksPerSortedBlock;

    // Reads sample value and calculates sample's global rank
    data_t sampleValue = keys[blockIdx.x * (blockDim.x * subBlockSize) + threadIdx.x * subBlockSize];
    uint_t rankSampleCurrent = blockIdx.x * blockDim.x + threadIdx.x;
    uint_t rankSampleOpposite;

    // Calculates index of current sorted block and opposite sorted block, with wich current block will be
    // merged (even - odd and vice versa)
    uint_t indexBlockCurrent = rankSampleCurrent / subBlocksPerSortedBlock;
    uint_t indexBlockOpposite = indexBlockCurrent ^ 1;

    // Searches for sample's rank in opposite block in order to calculate sample's index in merged block.
    // If current sample came from even block, it searches in corresponding odd block (and vice versa)
    if (indexBlockCurrent % 2 == 0)
    {
        rankSampleOpposite = binarySearchInclusive<sortOrder, subBlockSize>(
            keys, sampleValue, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
        );
        rankSampleOpposite = (rankSampleOpposite - sortedBlockSize) / subBlockSize;
    }
    else
    {
        rankSampleOpposite = binarySearchExclusive<sortOrder, subBlockSize>(
            keys, sampleValue, indexBlockOpposite * sortedBlockSize,
            indexBlockOpposite * sortedBlockSize + sortedBlockSize - subBlockSize
        );
        rankSampleOpposite /= subBlockSize;
    }

    // Calculates index of sample inside merged block
    uint_t sortedIndex = rankSampleCurrent % subBlocksPerSortedBlock + rankSampleOpposite;

    // Calculates sample's rank in current and opposite sorted block
    uint_t rankDataCurrent = (rankSampleCurrent * subBlockSize % sortedBlockSize) + 1;
    uint_t rankDataOpposite;

    // Calculate the index of sub-block within opposite sorted block
    uint_t indexSubBlockOpposite = sortedIndex % subBlocksPerMergedBlock - rankSampleCurrent % subBlocksPerSortedBlock - 1;
    // Start and end index for binary search
    uint_t indexStart = indexBlockOpposite * sortedBlockSize + indexSubBlockOpposite * subBlockSize + 1;
    uint_t indexEnd = indexStart + subBlockSize - 2;

    // Searches for sample's index in opposite sub-block (which is inside opposite sorted block)
    // Has to be explicitly converted to int, because it can be negative
    if ((int_t)(indexStart - indexBlockOpposite * sortedBlockSize) >= 0)
    {
        if (indexBlockOpposite % 2 == 0)
        {
            rankDataOpposite = binarySearchExclusive<sortOrder>(
                keys, sampleValue, indexStart, indexEnd
            );
        }
        else
        {
            rankDataOpposite = binarySearchInclusive<sortOrder>(
                keys, sampleValue, indexStart, indexEnd
            );
        }

        rankDataOpposite -= indexBlockOpposite * sortedBlockSize;
    }
    else
    {
        rankDataOpposite = 0;
    }

    // Outputs ranks
    if ((rankSampleCurrent / subBlocksPerSortedBlock) % 2 == 0)
    {
        ranksEven[sortedIndex] = rankDataCurrent;
        ranksOdd[sortedIndex] = rankDataOpposite;
    }
    else
    {
        ranksEven[sortedIndex] = rankDataOpposite;
        ranksOdd[sortedIndex] = rankDataCurrent;
    }
}

#endif
