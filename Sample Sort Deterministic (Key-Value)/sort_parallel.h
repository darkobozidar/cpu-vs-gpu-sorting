#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H


double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, data_t *d_samplesLocal, data_t *d_samplesGlobal,
    uint_t *d_localBucketSizes, uint_t *d_localBucketOffsets, uint_t *h_globalBucketOffsets,
    uint_t *d_globalBucketOffsets, uint_t tableLen, order_t sortOrder
);

#endif
