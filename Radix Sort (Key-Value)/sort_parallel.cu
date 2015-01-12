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
void runAddPaddingKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder)
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
        addPaddingKernel<MAX_VAL><<<dimGrid, dimBlock>>>(dataTable, tableLen, paddingLength);
    }
    else
    {
        addPaddingKernel<MIN_VAL><<<dimGrid, dimBlock>>>(dataTable, tableLen, paddingLength);
    }
}

/*
Runs kernel, which sorts data blocks in shared memory with radix sort according to current radix diggit,
which is specified with "bitOffset".
*/
void runRadixSortLocalKernel(data_t *keys, data_t *values, uint_t tableLen, uint_t bitOffset, order_t sortOrder)
{
    uint_t elemsPerThreadBlock = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    // "2" because of values, which are sorted alongside keys
    uint_t sharedMemSize = 2 * elemsPerThreadBlock * sizeof(*keys);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_LOCAL_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        radixSortLocalKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(keys, values, bitOffset);
    }
    else
    {
        radixSortLocalKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(keys, values, bitOffset);
    }
}

/*
Runs kernel, which generates local bucket offsets and sizes.
*/
void runGenerateBucketsKernel(
    data_t *dataTable, uint_t *blockOffsets, uint_t *blockSizes, uint_t tableLen, uint_t bitOffset
)
{
    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    // Shared memory size:
    // - "elemsPerLocalSort"  -> container for elements read from global memory into shared memory
    // - "2 * RADIX_PARALLEL" -> bucket local sizes + bucket local offsets
    uint_t sharedMemSize = elemsPerLocalSort * sizeof(uint_t) + 2 * RADIX_PARALLEL * sizeof(uint_t);

    dim3 dimGrid((tableLen - 1) / elemsPerLocalSort + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GEN_BUCKETS, 1, 1);

    generateBucketsKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
        dataTable, blockOffsets, blockSizes, bitOffset
    );
}

/*
Scatters elements to their corresponding buckets according to current radix diggit, which is specified
with "bitOffset".
*/
void runRadixSortGlobalKernel(
    data_t *dataKeys, data_t *dataValues, data_t *bufferKeys, data_t *bufferValues, uint_t *offsetsLocal,
    uint_t *offsetsGlobal, uint_t tableLen, uint_t bitOffset, order_t sortOrder
)
{
    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    // "2" because of values, which are sorted alongside keys
    uint_t sharedMemSIze = 2 * elemsPerLocalSort * sizeof(*dataKeys);

    dim3 dimGrid((tableLen - 1) / elemsPerLocalSort + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_GLOBAL_SORT, 1, 1);

    radixSortGlobalKernel<<<dimGrid, dimBlock, sharedMemSIze>>>(
        dataKeys, dataValues, bufferKeys, bufferValues, offsetsLocal, offsetsGlobal, bitOffset
    );
}

/*
Sorts data with parallel radix sort.
*/
double sortParallel(
    data_t *h_keys, data_t *h_values, data_t *d_dataKeys, data_t *d_dataValues, data_t *d_bufferKeys,
    data_t *d_bufferValues, uint_t *d_bucketOffsetsLocal, uint_t *d_bucketOffsetsGlobal, uint_t *d_bucketSizes,
    uint_t tableLen, order_t sortOrder
)
{
    uint_t elemsPerLocalSort = THREADS_PER_LOCAL_SORT * ELEMS_PER_THREAD_LOCAL;
    uint_t bucketsLen = RADIX_PARALLEL * ((tableLen - 1) / elemsPerLocalSort + 1);
    CUDPPHandle scanPlan;
    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    cudppInitScan(&scanPlan, bucketsLen);
    runAddPaddingKernel(d_dataKeys, tableLen, sortOrder);

    for (uint_t bitOffset = 0; bitOffset < sizeof(data_t) * 8; bitOffset += BIT_COUNT_PARALLEL)
    {
        runRadixSortLocalKernel(d_dataKeys, d_dataValues, tableLen, bitOffset, sortOrder);
        runGenerateBucketsKernel(d_dataKeys, d_bucketOffsetsLocal, d_bucketSizes, tableLen, bitOffset);

        // Performs global scan in order to calculate global bucket offsets from local bucket sizes
        CUDPPResult result = cudppScan(scanPlan, d_bucketOffsetsGlobal, d_bucketSizes, bucketsLen);
        if (result != CUDPP_SUCCESS)
        {
            printf("Error in cudppScan()\n");
            getchar();
            exit(-1);
        }

        runRadixSortGlobalKernel(
            d_dataKeys, d_dataValues, d_bufferKeys, d_bufferValues, d_bucketOffsetsLocal, d_bucketOffsetsGlobal,
            tableLen, bitOffset, sortOrder
        );

        data_t *temp = d_dataKeys;
        d_dataKeys = d_bufferKeys;
        d_bufferKeys = temp;

        temp = d_dataValues;
        d_dataValues = d_bufferValues;
        d_bufferValues = temp;
    }

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_keys, d_dataKeys, tableLen * sizeof(*h_keys), cudaMemcpyDeviceToHost);
    checkCudaError(error);
    error = cudaMemcpy(h_values, d_dataValues, tableLen * sizeof(*h_values), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
