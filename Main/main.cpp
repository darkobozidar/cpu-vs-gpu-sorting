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
#include "../Utils/generator.h"

#include "../Bitonic_Sort/sort_parallel.h"

#include "sort_test.h"


int main(int argc, char **argv)
{
    uint_t arrayLenStart = (1 << 20);
    uint_t arrayLenEnd = (1 << 20);
    uint_t interval = MAX_VAL;
    uint_t testRepetitions = 10;    // How many times are sorts ran
    order_t sortOrder = ORDER_ASC;  // Values: ORDER_ASC, ORDER_DESC

    // Input data distributions
    std::map<data_dist_t, std::string> mapDistribution = getDistributionMap();
    std::vector<data_dist_t> distributions;
    distributions.push_back(DISTRIBUTION_UNIFORM);
    distributions.push_back(DISTRIBUTION_GAUSSIAN);

    // Sorting algorithms
    std::vector<std::unique_ptr<Sort>> sortAlgorithms;
    sortAlgorithms.push_back(std::unique_ptr<Sort>(new BitonicSortParallelKeyOnly()));

    for (std::vector<data_dist_t>::iterator dist = distributions.begin(); dist != distributions.end(); dist++)
    {
        for (uint_t arrayLength = arrayLenStart; arrayLength <= arrayLenEnd; arrayLength *= 2)
        {
            data_t *keys = (data_t*)malloc(arrayLength * sizeof(*keys));
            data_t *values = (data_t*)malloc(arrayLength * sizeof(*values));

            for (uint_t iter = 0; iter < testRepetitions; iter++)
            {
                printf("> Test repetition: %d\n", iter);
                printf("> Distribution: %s\n", mapDistribution[*dist].c_str());
                printf("> Array length: %d\n", arrayLength);
                printf("> Data type: %s\n", typeid(data_t).name());

                printf("\n");
            }
        }

        printf("\n\n");
    }

    getchar();
    return 0;
}
