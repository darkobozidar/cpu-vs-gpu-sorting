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

#include "../BitonicSort/Sort/sequential.h"
#include "../BitonicSort/Sort/parallel.h"
#include "../BitonicSortMultistep/Sort/parallel.h"
#include "../BitonicSortAdaptive/Sort/sequential.h"
#include "../BitonicSortAdaptive/Sort/parallel.h"
#include "../MergeSort/Sort/sequential.h"
#include "../MergeSort/Sort/parallel.h"
#include "../Quicksort/Sort/sequential.h"
#include "../Quicksort/Sort/parallel.h"
#include "../RadixSort/Sort/sequential.h"
#include "../RadixSort/Sort/parallel.h"
#include "../SampleSort/Sort/sequential.h"
#include "../SampleSort/Sort/parallel.h"

#include "test_sort.h"


int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        printf(
            "Two mandatory and one optional argument has to be specified:\n1. array length\n2. number of test "
            "repetitions\n3. sort order (0 - ASC, 1 - DESC), optional, default ASC\n"
        );
        exit(EXIT_FAILURE);
    }

    uint_t arrayLength = atoi(argv[1]);
    // How many times is the sorting algorithm test repeated
    uint_t testRepetitions = atoi(argv[2]);
    // Sort order of the data
    order_t sortOrder = argc == 3 ? ORDER_ASC : (order_t)atoi(argv[3]);
    // Interval of input data -> [0, "interval]
    uint_t interval = MAX_VAL;

    // Input data distributions
    std::vector<data_dist_t> distributions;
    distributions.push_back(DISTRIBUTION_UNIFORM);
    distributions.push_back(DISTRIBUTION_GAUSSIAN);
    distributions.push_back(DISTRIBUTION_ZERO);
    distributions.push_back(DISTRIBUTION_BUCKET);
    distributions.push_back(DISTRIBUTION_SORTED_ASC);
    distributions.push_back(DISTRIBUTION_SORTED_DESC);

    // Sorting algorithms
    std::vector<SortSequential*> sorts;
    sorts.push_back(new BitonicSortSequential());
    sorts.push_back(new BitonicSortParallel());
    sorts.push_back(new BitonicSortMultistepParallel());
    sorts.push_back(new BitonicSortAdaptiveSequential());
    sorts.push_back(new BitonicSortAdaptiveParallel());
    sorts.push_back(new MergeSortSequential());
    sorts.push_back(new MergeSortParallel());
    sorts.push_back(new QuicksortSequential());
    sorts.push_back(new QuicksortParallel());
    sorts.push_back(new RadixSortSequential());
    sorts.push_back(new RadixSortParallel());
    sorts.push_back(new SampleSortSequential());
    sorts.push_back(new SampleSortParallel());

    // This is needed only for testing purposes, because data transfer from device to host shouldn't be timed.
    for (std::vector<SortSequential*>::iterator sort = sorts.begin(); sort != sorts.end(); sort++)
    {
        (*sort)->stopwatchEnable();
    }

    generateStatistics(sorts, distributions, arrayLength, sortOrder, testRepetitions, interval);

    return 0;
}
