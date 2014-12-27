#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <array>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


/*
From provided table offset, size of table block and length of entire table returns end index of the block.
*/
uint_t getEndIndex(uint_t offset, uint_t subBlockSize, uint_t tableLen)
{
    uint_t endIndex = offset + subBlockSize;
    return endIndex <= tableLen ? endIndex : tableLen;
}


/*
Sorts data sequentially with merge sort.

Pointers to data table and data buffer are carried by reference in order to insure that output data in always
in primary array. If it wasn't for this, there would be 50% chance (depending on array size) that output would
be in buffer array.
*/
double sortSequential(
    data_t *&dataKeys, data_t *&dataValues, data_t *&keysBuffer, data_t *&valuesBuffer, uint_t tableLen,
    order_t sortOrder
)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    uint_t tableLenPower2 = nextPowerOf2(tableLen);

    // Log(tableLen) phases of merge sort
    for (uint_t sortedBlockSize = 2; sortedBlockSize <= tableLenPower2; sortedBlockSize *= 2)
    {
        // Number of merged blocks that will be created in this iteration
        uint_t numBlocks = (tableLen - 1) / sortedBlockSize + 1;
        // Number of sub-blocks being merged
        uint_t subBlockSize = sortedBlockSize / 2;

        // Merge of all blocks
        for (uint_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
        {
            // Odd (left) block being merged
            uint_t oddIndex = blockIndex * sortedBlockSize;
            uint_t oddEnd = getEndIndex(oddIndex, subBlockSize, tableLen);

            // If there is only odd block without even block, then only odd block is coppied into buffer
            if (oddEnd == tableLen)
            {
                std::copy(dataKeys + oddIndex, dataKeys + oddEnd, keysBuffer + oddIndex);
                continue;
            }

            // Even (right) block being merged
            uint_t evenIndex = oddIndex + subBlockSize;
            uint_t evenEnd = getEndIndex(evenIndex, subBlockSize, tableLen);
            uint_t mergeIndex = oddIndex;

            // Merge of odd and even block
            while (oddIndex < oddEnd && evenIndex < evenEnd)
            {
                data_t oddElement = dataKeys[oddIndex];
                data_t evenElement = dataKeys[evenIndex];

                if (sortOrder == ORDER_ASC ? oddElement < evenElement : oddElement > evenElement)
                {
                    keysBuffer[mergeIndex++] = oddElement;
                    oddIndex++;
                }
                else
                {
                    keysBuffer[mergeIndex++] = evenElement;
                    evenIndex++;
                }
            }

            // Block that wasn't merged entirely is coppied into buffer array
            if (oddIndex == oddEnd)
            {
                std::copy(dataKeys + evenIndex, dataKeys + evenEnd, keysBuffer + mergeIndex);
            }
            else
            {
                std::copy(dataKeys + oddIndex, dataKeys + oddEnd, keysBuffer + mergeIndex);
            }
        }

        // Exchanges key and value pointers with buffer
        data_t *temp = dataKeys;
        dataKeys = keysBuffer;
        keysBuffer = temp;
    }

    return endStopwatch(timer);
}
