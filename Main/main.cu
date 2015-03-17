#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <array>
#include <vector>
#include <memory>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Utils/data_types_common.h"
#include "../Utils/host.h"
#include "../Utils/cuda.h"
#include "../Utils/sort_interface.h"

#include "../BitonicSort/sort_sequential.h"
#include "../BitonicSort/sort_parallel.h"
#include "../BitonicSortMultistep/sort_parallel.h"
#include "../BitonicSortAdaptive/sort_sequential.h"
#include "../BitonicSortAdaptive/sort_parallel.h"

#include "test_sort.h"


int main(int argc, char **argv)
{
    uint_t arrayLenStart = (1 << 20);
    uint_t arrayLenEnd = arrayLenStart;
    uint_t interval = MAX_VAL;
    uint_t testRepetitions = 3;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC

    // Input data distributions
    std::vector<data_dist_t> distributions;
    distributions.push_back(DISTRIBUTION_UNIFORM);

    // Sorting algorithms
    std::vector<SortSequential*> sorts;
    sorts.push_back(new BitonicSortSequential());
    sorts.push_back(new BitonicSortParallel());
    sorts.push_back(new BitonicSortMultistepParallel());
    sorts.push_back(new BitonicSortAdaptiveSequential());
    sorts.push_back(new BitonicSortAdaptiveParallel());
    //sorts.push_back(new QuicksortSequential());
    //sorts.push_back(new QuicksortParallel());

    // This is needed only for testing puproses, because data transfer from device to host shouldn't be stopwatched.
    for (std::vector<SortSequential*>::iterator sort = sorts.begin(); sort != sorts.end(); sort++)
    {
        (*sort)->stopwatchEnable();
    }

    generateStatistics(sorts, distributions, arrayLenStart, arrayLenEnd, sortOrder, testRepetitions, interval);

    printf("Finished\n");
    getchar();
    return 0;
}
