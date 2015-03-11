#ifndef QUICKSORT_PARALLEL_KEY_VALUE_H
#define QUICKSORT_PARALLEL_KEY_VALUE_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "data_types.h"


class QuicksortParallelKeyValue : public SortParallelKeyValue
{
private:
    // Buffer for keys and values
    data_t *d_keysBuffer, *d_valuesBuffer;
    // When pivots are scattered in global and local quicksort, they have to be considered as unique elements
    // because of array of values (alongside keys). Because array can contain duplicate keys, values have to
    // be stored in buffer, because end position of pivots isn't known until last thread block processes sequence.
    data_t *d_pivotValues;
    // When initial min/max parallel reduction reduces data to threashold, min/max values are coppied to host
    // and reduction is finnished on host. Multiplier "2" is used because of min and max values.
    data_t *h_minMaxValues;
    // Sequences metadata for GLOBAL quicksort on HOST
    h_glob_seq_t *h_globalSeqHost, *h_globalSeqHostBuffer;
    // Sequences metadata for GLOBAL quicksort on DEVICE
    d_glob_seq_t *h_globalSeqDev, *d_globalSeqDev;
    // Array of sequence indexes for thread blocks in GLOBAL quicksort. This way thread blocks know which
    // sequence they have to partition.
    uint_t *h_globalSeqIndexes, *d_globalSeqIndexes;
    // Sequences metadata for LOCAL quicksort
    loc_seq_t *h_localSeq, *d_localSeq;
    std::string sortName = "Quicksort parallel key value";

    void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    void memoryDestroy();
    void memoryCopyDeviceToHost(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    uint_t runMinMaxReductionKernel();
    void minMaxReduction(data_t &minVal, data_t &maxVal);
    void runQuickSortGlobalKernel(uint_t numSeqGlobal, uint_t threadBlockCounter);
    void runQuickSortLocalKernel(uint_t numThreadBlocks);
    void sortPrivate();

public:
    std::string getSortName()
    {
        return this->sortName;
    }
};

#endif
