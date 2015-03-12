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

    // Key only
    void sortKeyOnly();

    // Key-value
    void sortKeyValue();

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};

#endif
