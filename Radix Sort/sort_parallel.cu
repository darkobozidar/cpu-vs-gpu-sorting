#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudpp.h>

#include "../Utils/data_types_common.h"
#include "../Utils/cuda.h"
#include "../Utils/host.h"
#include "constants.h"
#include "kernels.h"


/*
Initializes library CUDPP, which implements scan() function
*/
void cudppInitScan(CUDPPHandle *scanPlan, uint_t tableLen)
{
    // Initializes the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    *scanPlan = 0;
    CUDPPResult result = cudppPlan(theCudpp, scanPlan, config, tableLen, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error creating CUDPPPlan\n");
        getchar();
        exit(-1);
    }
}

/*
Adds padding of MAX/MIN values to input table, deppending if sort order is ascending or descending. This is
needed, if table length is not divisable with number of elements sorted by local radix sort. In order for
parallel sort to work, table length has to be divisable with number of elements processed by one thread block
in local radix sort.
*/
void runAddPaddingKernel(data_t *dataTable, data_t *dataBuffer, uint_t tableLen, order_t sortOrder)
{
    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    uint_t tableLenRoundedUp = roundUp(tableLen, elemsPerLocalSort);

    // If table length is already power of 2, than no padding is needed
    if (tableLen == tableLenRoundedUp)
    {
        return;
    }

    uint_t paddingLength = tableLenRoundedUp - tableLen;

    uint_t elemsPerThreadBlock = THREADS_PER_PADDING * ELEMS_PER_THREAD_PADDING;;
    dim3 dimGrid((paddingLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_PADDING, 1, 1);

    // Depending on sort order different value is used for padding.
    if (sortOrder == ORDER_ASC)
    {
        addPaddingKernel<MAX_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
    else
    {
        addPaddingKernel<MIN_VAL><<<dimGrid, dimBlock>>>(dataTable, dataBuffer, tableLen, paddingLength);
    }
}

/*
Runs kernel, which sorts data blocks in shared memory with radix sort according to current radix diggit,
which is specified with "bitOffset".
*/
void runRadixSortLocalKernel(data_t *dataTable, uint_t tableLen, uint_t bitOffset, order_t sortOrder)
{
    uint_t threadBlockSize = min((tableLen - 1) / ELEMS_PER_THREAD_LOCAL + 1, THREADS_PER_LOCAL_SORT);
    uint_t sharedMemSize = threadBlockSize * ELEMS_PER_THREAD_LOCAL * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / (ELEMS_PER_THREAD_LOCAL * threadBlockSize) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        radixSortLocalKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, bitOffset
        );
    }
    else
    {
        radixSortLocalKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(
            dataTable, bitOffset
        );
    }
}

void runGenerateBucketsKernel(
    data_t *dataTable, uint_t *blockOffsets, uint_t *blockSizes, uint_t tableLen, uint_t bitOffset
)
{
    uint_t threadBlockSize = min((tableLen - 1) / ELEMS_PER_THREAD_LOCAL + 1, THREADS_PER_LOCAL_SORT);
    uint_t sharedMemSize = ELEMS_PER_THREAD_LOCAL * threadBlockSize * sizeof(uint_t) + 2 * RADIX_PARALLEL * sizeof(uint_t);

    dim3 dimGrid((tableLen - 1) / (ELEMS_PER_THREAD_LOCAL * threadBlockSize) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateBucketsKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataTable, blockOffsets, blockSizes, bitOffset
    );
}

void runRadixSortGlobalKernel(
    data_t *dataTable,  data_t *dataBuffer, uint_t *offsetsLocal, uint_t *offsetsGlobal, uint_t tableLen,
    uint_t bitOffset, order_t sortOrder
)
{
    uint_t threadBlockSize = min((tableLen - 1) / ELEMS_PER_THREAD_LOCAL, THREADS_PER_GLOBAL_SORT);
    uint_t sharedMemSIze = ELEMS_PER_THREAD_LOCAL * threadBlockSize * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / (ELEMS_PER_THREAD_LOCAL * threadBlockSize) + 1, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    radixSortGlobalKernel<<<dimGrid, dimBlock, sharedMemSIze>>>(
        dataTable, dataBuffer, offsetsLocal, offsetsGlobal, bitOffset
    );
}

//void runPrintTableKernel(uint_t *table, uint_t tableLen) {
//    printTableKernel<<<1, 1>>>(table, tableLen);
//    cudaError_t error = cudaDeviceSynchronize();
//    checkCudaError(error);
//}

double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, uint_t *d_bucketOffsetsLocal,
    uint_t *d_bucketOffsetsGlobal, uint_t *d_bucketSizes, uint_t tableLen, order_t sortOrder
)
{
    uint_t threadsPerSortLocal = min((tableLen - 1) / ELEMS_PER_THREAD_LOCAL + 1, THREADS_PER_LOCAL_SORT);
    uint_t bucketsLen = RADIX_PARALLEL * (tableLen / (threadsPerSortLocal * ELEMS_PER_THREAD_LOCAL));
    CUDPPHandle scanPlan;
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    cudppInitScan(&scanPlan, bucketsLen);
    runAddPaddingKernel(d_dataTable, d_dataBuffer, tableLen, sortOrder);

    for (uint_t bitOffset = 0; bitOffset < sizeof(data_t) * 8; bitOffset += BIT_COUNT_PARALLEL)
    {
        runRadixSortLocalKernel(d_dataTable, tableLen, bitOffset, sortOrder);
        runGenerateBucketsKernel(d_dataTable, d_bucketOffsetsLocal, d_bucketSizes, tableLen, bitOffset);

        CUDPPResult result = cudppScan(scanPlan, d_bucketOffsetsGlobal, d_bucketSizes, bucketsLen);
        if (result != CUDPP_SUCCESS)
        {
            printf("Error in cudppScan()\n");
            getchar();
            exit(-1);
        }

        runRadixSortGlobalKernel(
            d_dataTable, d_dataBuffer, d_bucketOffsetsLocal, d_bucketOffsetsGlobal, tableLen, bitOffset, sortOrder
        );

        data_t *temp = d_dataTable;
        d_dataTable = d_dataBuffer;
        d_dataBuffer = temp;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
