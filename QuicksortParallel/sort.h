#ifndef QUICKSORT_PARALLEL_H
#define QUICKSORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "data_types.h"


/*
Due to (extreme =)) optimization code for key only and key-value sorts are entirely separated.
TODO once the testing is done merge common code for key only and key-value sorts.
*/
class QuicksortParallel : public SortParallel
{
private:
    std::string _sortName = "Quicksort parallel";
    // Device buffer for keys and values
    data_t *_d_keysBuffer, *_d_valuesBuffer;
    // When pivots are scattered in global and local quicksort, they have to be considered as unique elements
    // because of array of values (alongside keys). Because array can contain duplicate keys, values have to
    // be stored in buffer, because end position of pivots isn't known until last thread block processes sequence.
    data_t *_d_valuesPivot;
    // When initial min/max parallel reduction reduces data to threashold, min/max values are coppied to host
    // and reduction is finnished on host. Multiplier "2" is used because of min and max values.
    data_t *_h_minMaxValues;
    // Sequences metadata for GLOBAL quicksort on HOST
    h_glob_seq_t *_h_globalSeqHost, *_h_globalSeqHostBuffer;
    // Sequences metadata for GLOBAL quicksort on DEVICE
    d_glob_seq_t *_h_globalSeqDev, *_d_globalSeqDev;
    // Array of sequence indexes for thread blocks in GLOBAL quicksort. This way thread blocks know which
    // sequence they have to partition.
    uint_t *_h_globalSeqIndexes, *_d_globalSeqIndexes;
    // Sequences metadata for LOCAL quicksort
    loc_seq_t *_h_localSeq, *_d_localSeq;

    // Common
    void memoryAllocate(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    void memoryDestroy();
    void memoryCopyAfterSort(data_t *h_keys, data_t *h_values, uint_t arrayLength);
    template <uint_t threadsReduction, uint_t elemsThreadReduction>
    uint_t runMinMaxReductionKernel(data_t *d_keys, data_t *d_keysBuffer, uint_t arrayLength);
    template <uint_t thresholdReduction, uint_t threadsReduction, uint_t elemsThreadReduction>
    void minMaxReduction(
        data_t *h_keys, data_t *d_keys, data_t *d_keysBuffer, data_t *h_minMaxValues, uint_t arrayLength,
        data_t &minVal, data_t &maxVal
    );

    // Key only
    void runQuickSortGlobalKernel(
        data_t *d_keys, data_t *d_keysBuffer, d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev,
        uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, uint_t numSeqGlobal, uint_t threadBlockCounter
    );
    template <order_t sortOrder>
    void runQuickSortLocalKernel(
        data_t *d_keys, data_t *d_keysBuffer, loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t numThreadBlocks
    );
    template <order_t sortOrder>
    void quicksortParallel(
        data_t *h_keys, data_t *&d_keys, data_t *&d_keysBuffer, data_t *h_minMaxValues,
        h_glob_seq_t *h_globalSeqHost, h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev,
        d_glob_seq_t *d_globalSeqDev, uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes,
        loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t arrayLength
    );
    void sortKeyOnly();

    // Key-value
    void runQuickSortGlobalKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, data_t *d_valuesPivot,
        d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev, uint_t *h_globalSeqIndexes,
        uint_t *d_globalSeqIndexes, uint_t numSeqGlobal, uint_t threadBlockCounter
    );
    template <order_t sortOrder>
    void runQuickSortLocalKernel(
        data_t *d_keys, data_t *d_values, data_t *d_keysBuffer, data_t *d_valuesBuffer, data_t *d_valuesPivot,
        loc_seq_t *h_localSeq, loc_seq_t *d_localSeq, uint_t numThreadBlocks
    );
    template <order_t sortOrder>
    void quicksortParallel(
        data_t *h_keys, data_t *&d_keys, data_t *&d_values, data_t *&d_keysBuffer, data_t *&d_valuesBuffer,
        data_t *d_valuesPivot, data_t *h_minMaxValues, h_glob_seq_t *h_globalSeqHost,
        h_glob_seq_t *h_globalSeqHostBuffer, d_glob_seq_t *h_globalSeqDev, d_glob_seq_t *d_globalSeqDev,
        uint_t *h_globalSeqIndexes, uint_t *d_globalSeqIndexes, loc_seq_t *h_localSeq, loc_seq_t *d_localSeq,
        uint_t arrayLength
    );
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
