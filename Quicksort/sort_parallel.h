#ifndef QUICKSORT_PARALLEL_H
#define QUICKSORT_PARALLEL_H

#include "../Utils/data_types_common.h"
#include "../Utils/sort_interface.h"
#include "constants.h"
#include "data_types.h"


/*
Base class for parallel bitonic sort.
Needed for template specialization.

Template params:
_Ko - Key-only
_Kv - Key-value

TODO implement DESC ordering.
*/
template <
    uint_t useReductionInGlobalSort, uint_t thresholdParallelReduction,
    uint_t threadsReduction, uint_t elemsReduction,
    uint_t threasholdPartitionGlobalKo, uint_t threasholdPartitionGlobalKv,
    uint_t threadsSortGlobalKo, uint_t elemsSortGlobalKo,
    uint_t threadsSortGlobalKv, uint_t elemsSortGlobalKv,
    uint_t thresholdBitonicSortKo, uint_t thresholdBitonicSortKv,
    uint_t threadsSortLocalKo, uint_t threadsSortLocalKv
>
class QuicksortParallelBase : public SortParallel
{
protected:
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

public:
    std::string getSortName()
    {
        return this->_sortName;
    }
};


/*
Class for parallel quicksort.
*/
class QuicksortParallel : public QuicksortParallelBase<
    USE_REDUCTION_IN_GLOBAL_SORT, THRESHOLD_PARALLEL_REDUCTION,
    THREADS_PER_REDUCTION, ELEMENTS_PER_THREAD_REDUCTION,
    THRESHOLD_PARTITION_SIZE_GLOBAL_KO, THRESHOLD_PARTITION_SIZE_GLOBAL_KV,
    THREADS_PER_SORT_GLOBAL_KO, ELEMENTS_PER_THREAD_GLOBAL_KO,
    THREADS_PER_SORT_GLOBAL_KV, ELEMENTS_PER_THREAD_GLOBAL_KV,
    THRESHOLD_BITONIC_SORT_KO, THRESHOLD_BITONIC_SORT_KV,
    THREADS_PER_SORT_LOCAL_KO, THREADS_PER_SORT_LOCAL_KV
>
{};

#endif
