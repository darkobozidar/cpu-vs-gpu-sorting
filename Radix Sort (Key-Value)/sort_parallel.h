#ifndef SORT_PARALLEL_H
#define SORT_PARALLEL_H


double sortParallel(
    data_t *h_output, data_t *d_dataTable, data_t *d_dataBuffer, uint_t *d_bucketOffsetsLocal,
    uint_t *d_bucketOffsetsGlobal, uint_t *d_bucketSizes, uint_t tableLen, order_t sortOrder
);

#endif
