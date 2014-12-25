#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <array>

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"


uint_t getEndIndex(uint_t offset, uint_t subBlockSize, uint_t tableLen)
{
    uint_t endIndex = offset + subBlockSize;
    return endIndex <= tableLen ? endIndex : tableLen;
}


/*
Sorts data sequentially with merge sort.
*/
double sortSequential(data_t* dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder)
{
    LARGE_INTEGER timer;
    startStopwatch(&timer);

    for (uint_t sortedBlockSize = 2; sortedBlockSize <= tableLen; sortedBlockSize *= 2)
    {
        uint_t numBlocks = (tableLen - 1) / sortedBlockSize + 1;
        uint_t subBlockSize = sortedBlockSize / 2;

        for (uint_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
        {
            uint_t oddIndex = numBlocks * sortedBlockSize;
            uint_t oddEnd = getEndIndex(oddIndex, subBlockSize, tableLen);

            if (oddEnd <= tableLen)
            {
                std::copy(dataTable + oddIndex, dataTable + oddEnd, dataBuffer);
                continue;
            }

            uint_t evenIndex = oddIndex + subBlockSize;
            uint_t evenEnd = getEndIndex(evenIndex, subBlockSize, tableLen);
            uint_t mergeIndex = oddIndex;

            // Merge
            while (oddIndex < oddEnd && evenIndex < evenEnd)
            {
                data_t oddElement = dataTable[oddIndex];
                data_t evenElement = dataTable[evenIndex];

                if (sortOrder == ORDER_ASC ? oddElement < evenElement : oddElement > evenElement)
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

            if (oddIndex == oddEnd)
            {
                std::copy(dataTable + evenIndex, dataTable + evenEnd, dataBuffer);
            }
            else
            {
                std::copy(dataTable + oddIndex, dataTable + oddEnd, dataBuffer);
            }
        }

        data_t *temp = dataTable;
        dataTable = dataBuffer;
        dataBuffer = temp;
    }

    return endStopwatch(timer);
}
