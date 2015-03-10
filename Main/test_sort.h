#ifndef TEST_SORT_H
#define TEST_SORT_H

#include <vector>
#include <memory>

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"


void testSorts(
    std::vector<Sort*> sorts, std::vector<data_dist_t> distributions, uint_t arrayLenStart, uint_t arrayLenEnd,
    uint_t testRepetitions, uint_t interval
);

#endif
