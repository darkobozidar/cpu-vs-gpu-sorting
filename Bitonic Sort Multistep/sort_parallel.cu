#include <stdio.h>
#include <math.h>
#include <Windows.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "constants.h"
#include "kernels.h"


/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitoicSortKernel(data_t *dataTable, uint_t tableLen, order_t sortOrder) {
    uint_t elemsPerThreadBlock = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    uint_t sharedMemSize = elemsPerThreadBlock * sizeof(*dataTable);

    dim3 dimGrid((tableLen - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_PER_BITONIC_SORT, 1, 1);

    if (sortOrder == ORDER_ASC)
    {
        bitonicSortKernel<ORDER_ASC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable, tableLen);
    }
    else
    {
        bitonicSortKernel<ORDER_DESC><<<dimGrid, dimBlock, sharedMemSize>>>(dataTable, tableLen);
    }
}

///*
//Runs multistep kernel, which uses registers.
//*/
//void runMultiStepKernel(el_t *table, uint_t tableLen, uint_t phase, uint_t step, uint_t degree,
//                                 bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    uint_t partitionSize = tableLen / (1 << degree);
//    uint_t maxThreadBlockSize = MAX_THREADS_PER_MULTISTEP;
//    uint_t threadBlockSize = min(partitionSize, maxThreadBlockSize);
//    dim3 dimGrid(partitionSize / threadBlockSize, 1, 1);
//    dim3 dimBlock(threadBlockSize, 1, 1);
//
//    startStopwatch(&timer);
//    if (degree == 1) {
//        multiStep1Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
//    } else if (degree == 2) {
//        multiStep2Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
//    } else if (degree == 3) {
//        multiStep3Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
//    } else if (degree == 4) {
//        multiStep4Kernel<<<dimGrid, dimBlock>>>(table, phase, step, orderAsc);
//    }
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing multistep kernel using registers");*/
//}
//
//void runBitoicMergeKernel(el_t *table, uint_t tableLen, uint_t phasesBitonicMerge, uint_t phase, bool orderAsc) {
//    cudaError_t error;
//    LARGE_INTEGER timer;
//
//    // Every thread loads and sorts 2 elements
//    uint_t subBlockSize = 1 << (phasesBitonicMerge + 1);
//    dim3 dimGrid(tableLen / subBlockSize, 1, 1);
//    dim3 dimBlock(subBlockSize / 2, 1, 1);
//
//    startStopwatch(&timer);
//    bitonicMergeKernel<<<dimGrid, dimBlock, subBlockSize * sizeof(*table)>>>(
//        table, phase, orderAsc
//    );
//    /*error = cudaDeviceSynchronize();
//    checkCudaError(error);
//    endStopwatch(timer, "Executing bitonic sort kernel");*/
//}

/*
Sorts data with NORMALIZED BITONIC SORT.
*/
double sortParallel(data_t *h_output, data_t *d_dataTable, uint_t tableLen, order_t sortOrder)
{
    uint_t tableLenPower2 = nextPowerOf2(tableLen);
    uint_t elemsPerBlockBitonicSort = THREADS_PER_BITONIC_SORT * ELEMS_PER_THREAD_BITONIC_SORT;
    /*uint_t elemsPerBlockMergeLocal = THREADS_PER_LOCAL_MERGE * ELEMS_PER_THREAD_LOCAL_MERGE;*/

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    uint_t phasesBitonicSort = log2((double)min(tableLenPower2, elemsPerBlockBitonicSort));
    /*uint_t phasesMergeLocal = log2((double)min(tableLenPower2, elemsPerBlockMergeLocal));*/
    uint_t phasesAll = log2((double)tableLenPower2);

    LARGE_INTEGER timer;
    cudaError_t error;

    startStopwatch(&timer);
    runBitoicSortKernel(d_dataTable, tableLen, sortOrder);

    //// Bitonic merge
    //for (uint_t phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    //{
    //    uint_t step = phase;
    //    while (step > phasesMergeLocal)
    //    {
    //        runBitonicMergeGlobalKernel(d_dataTable, tableLen, phase, step, sortOrder);
    //        step--;
    //    }

    //    runBitoicMergeLocalKernel(d_dataTable, tableLen, phase, step, sortOrder);
    //}

    error = cudaDeviceSynchronize();
    checkCudaError(error);
    double time = endStopwatch(timer);

    error = cudaMemcpy(h_output, d_dataTable, tableLen * sizeof(*h_output), cudaMemcpyDeviceToHost);
    checkCudaError(error);

    return time;
}
