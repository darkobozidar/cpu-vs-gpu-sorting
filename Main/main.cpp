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

#include "../Bitonic_Sort/sort_parallel.h"

#include "test_sort.h"


int main(int argc, char **argv)
{
    uint_t arrayLenStart = (1 << 5);
    uint_t arrayLenEnd = (1 << 10);
    uint_t interval = MAX_VAL;
    uint_t testRepetitions = 10;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC

    // Input data distributions
    std::vector<data_dist_t> distributions;
    distributions.push_back(DISTRIBUTION_UNIFORM);

    // Sorting algorithms
    std::vector<Sort*> sortAlgorithms;
    sortAlgorithms.push_back(new BitonicSortParallelKeyOnly());

    // This is needed only for testing puproses, because data transfer from device to host shouldn't be stopwatched.
    for (std::vector<Sort*>::iterator sort = sortAlgorithms.begin(); sort != sortAlgorithms.end(); sort++)
    {
        (*sort)->setMemoryCopyAfterSort(false);
    }

    testSorts(sortAlgorithms, distributions, arrayLenStart, arrayLenEnd, sortOrder, testRepetitions, interval);

    printf("Finished\n");
    getchar();
    return 0;
}
