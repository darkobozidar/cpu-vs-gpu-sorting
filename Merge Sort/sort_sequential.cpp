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
double sortSequential(data_t *&dataTable, data_t *&dataBuffer, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    // Log(tableLen) phases of merge sort
    for (uint_t subBlockSize = 1; subBlockSize < tableLen; subBlockSize *= 2)
    {
        // Number of merged blocks that will be created in this iteration
        uint_t numBlocks = (tableLen - 1) / (2 * subBlockSize) + 1;

        // Merge of all blocks
        for (uint_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
        {
            // Odd (left) block being merged
            uint_t oddIndex = blockIndex * (2 * subBlockSize);
            uint_t oddEnd = getEndIndex(oddIndex, subBlockSize, tableLen);
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
                    dataBuffer[mergeIndex++] = oddElement;
                    oddIndex++;
                }
                else
                {
                    dataBuffer[mergeIndex++] = evenElement;
                    evenIndex++;
                }
            }

            // Block that wasn't merged entirely is coppied into buffer array
            if (oddIndex >= oddEnd && evenIndex < evenEnd)
            {
                std::copy(dataTable + evenIndex, dataTable + evenEnd, dataBuffer + mergeIndex);
            }
            else
            {
                std::copy(dataTable + oddIndex, dataTable + oddEnd, dataBuffer + mergeIndex);
            }
        }

        data_t *temp = dataTable;
        dataTable = dataBuffer;
        dataBuffer = temp;
    }

    return endStopwatch(timer);
}
